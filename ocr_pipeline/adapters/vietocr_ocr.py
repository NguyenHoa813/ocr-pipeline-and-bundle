from __future__ import annotations

import re
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VIETOCR_CACHE_DIR = PROJECT_ROOT / "cache" / "vietocr"
VIETOCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)

from PIL import Image, UnidentifiedImageError

from ocr_pipeline.adapters.preprocess import (
    PreparedPage,
    prepare_page,
)
from ocr_pipeline.adapters.tesseract_ocr import (
    TESSERACT_CONFIG,
    TESSERACT_MIN_TEXT_SIDE,
)
from ocr_pipeline.adapters.text_layout import (
    TextLine,
    is_noise_text,
    markdown_from_paragraphs,
    normalize_box,
    normalize_text,
)
from ocr_pipeline.steps.step4_ocr_pages import (
    LayoutRegion,
    PageOCRContent,
    Step4Error,
)

_LANGUAGES = {
    "vi": "vie",
    "vi+zh-Hant": "vie+chi_tra",
}
_WEIGHTS = {
    "vgg_transformer": "https://vocr.vn/data/vietocr/vgg_transformer.pth",
    "vgg_seq2seq": "https://vocr.vn/data/vietocr/vgg_seq2seq.pth",
}
_CHINESE_CONFIG = "--oem 1 --psm 7 -c preserve_interword_spaces=1"
_CJK_PATTERN = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]"
)
_MATH_SYMBOLS = set("+=<>±×÷√∑∏∫≈≠≤≥")


@dataclass(frozen=True)
class _Line:
    key: tuple[int, int, int]
    paragraph: tuple[int, int]
    bbox: tuple[float, float, float, float]
    tesseract_text: str


def _cjk_count(text: str) -> int:
    return len(_CJK_PATTERN.findall(text))


def _cjk_share(text: str) -> float:
    visible = [character for character in text if not character.isspace()]
    if not visible:
        return 0.0
    return _cjk_count(text) / len(visible)


def _alphabetic(token: str) -> str:
    return "".join(character for character in token if character.isalpha())


def _merge_symbol_tokens(recognized: str, tesseract: str) -> str:
    """Keep VietOCR words, restoring case and punctuation Tesseract read."""
    recognized_tokens = recognized.split()
    tesseract_tokens = tesseract.split()
    if not recognized_tokens:
        return tesseract
    if len(recognized_tokens) == len(tesseract_tokens):
        merged: list[str] = []
        for recognized_token, tesseract_token in zip(
            recognized_tokens,
            tesseract_tokens,
        ):
            recognized_letters = _alphabetic(recognized_token)
            tesseract_letters = _alphabetic(tesseract_token)
            if (
                recognized_letters
                and recognized_letters.casefold()
                == tesseract_letters.casefold()
                and any(
                    not character.isalnum() for character in tesseract_token
                )
                and not any(
                    character in "®©¬�"
                    for character in tesseract_token
                )
            ):
                merged.append(tesseract_token)
            elif (
                tesseract_token.isupper()
                and recognized_token[:1].isupper()
                and len(tesseract_letters) >= 2
            ):
                merged.append(recognized_token.upper())
            elif (
                not any(character.isalnum() for character in recognized_token)
                and any(
                    character in _MATH_SYMBOLS
                    for character in tesseract_token
                )
            ):
                merged.append(tesseract_token)
            else:
                merged.append(recognized_token)
        return " ".join(merged)
    if len(tesseract_tokens) == len(recognized_tokens) + 1:
        extra = tesseract_tokens[-1]
        if any(character in _MATH_SYMBOLS for character in extra):
            return " ".join([*recognized_tokens, extra])
    return recognized


def _ensure_weights(variant: str) -> Path:
    destination = VIETOCR_CACHE_DIR / f"{variant}.pth"
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    url = _WEIGHTS.get(variant)
    if url is None:
        raise Step4Error(f"Unknown VietOCR model variant: {variant}.")
    temporary = destination.with_suffix(".pth.part")
    try:
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(destination)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise Step4Error(
            "Cannot download the VietOCR weights. Download them manually "
            f"from {url} to {destination}. Details: {exc}"
        ) from exc
    return destination


def _line_boxes(data: dict[str, Any]) -> list[_Line]:
    required = (
        "text",
        "left",
        "top",
        "width",
        "height",
        "block_num",
        "par_num",
        "line_num",
    )
    if any(not isinstance(data.get(field), (list, tuple)) for field in required):
        raise Step4Error("Tesseract returned an invalid TSV result.")
    if len({len(data[field]) for field in required}) != 1:
        raise Step4Error("Tesseract returned mismatched TSV columns.")

    grouped: "OrderedDict[tuple[int, int, int], list[Any]]" = OrderedDict()
    for index, value in enumerate(data["text"]):
        text = value.strip() if isinstance(value, str) else ""
        if not text:
            continue
        key = (
            int(data["block_num"][index]),
            int(data["par_num"][index]),
            int(data["line_num"][index]),
        )
        left = int(data["left"][index])
        top = int(data["top"][index])
        width = int(data["width"][index])
        height = int(data["height"][index])
        if width <= 0 or height <= 0:
            continue
        if key not in grouped:
            grouped[key] = [
                [left, top, left + width, top + height],
                [text],
            ]
        else:
            box, texts = grouped[key]
            box[0] = min(box[0], left)
            box[1] = min(box[1], top)
            box[2] = max(box[2], left + width)
            box[3] = max(box[3], top + height)
            texts.append(text)

    lines: list[_Line] = []
    for key, (box, texts) in grouped.items():
        lines.append(
            _Line(
                key=key,
                paragraph=key[:2],
                bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                tesseract_text=normalize_text(" ".join(texts)),
            )
        )
    return lines


class VietOCRAdapter:
    """Line-level Vietnamese recognizer with optional Traditional Chinese."""

    model_name = "VietOCR"
    model_version = "VietOCR"

    def __init__(
        self,
        language: str = "vi",
        preprocess: bool = True,
        min_text_side: int = TESSERACT_MIN_TEXT_SIDE,
        variant: str = "vgg_transformer",
        weights_path: Path | None = None,
    ) -> None:
        if language not in _LANGUAGES:
            raise Step4Error(
                "VietOCR language must be one of: "
                + ", ".join(_LANGUAGES)
                + "."
            )
        if variant not in _WEIGHTS:
            raise Step4Error(f"Unknown VietOCR model variant: {variant}.")
        try:
            import pytesseract
            import vietocr
            from vietocr.tool.config import Cfg
            from vietocr.tool.predictor import Predictor
        except ImportError as exc:
            raise Step4Error(
                "VietOCR needs its runtime. Install it without replacing the "
                "existing Pillow/OpenCV stack:\n"
                "  python -m pip install --no-deps vietocr==0.3.13\n"
                "  python -m pip install gdown einops prefetch-generator lmdb\n"
                f"Details: {exc}"
            ) from exc

        self.language = language
        self.detection_language = _LANGUAGES[language]
        self.preprocess = preprocess
        self.min_text_side = min_text_side
        self.variant = variant
        self._pytesseract = pytesseract
        self._weights_path = Path(weights_path) if weights_path else _ensure_weights(
            variant
        )
        try:
            config = Cfg.load_config_from_name(variant)
            config["weights"] = str(self._weights_path)
            config["device"] = "cpu"
            config["cnn"]["pretrained"] = False
            self._predictor = Predictor(config)
            self._tesseract_version = str(pytesseract.get_tesseract_version())
            try:
                vietocr_version = package_version("vietocr")
            except PackageNotFoundError:
                vietocr_version = getattr(vietocr, "__version__", "unknown")
        except Exception as exc:
            raise Step4Error(f"Cannot initialize VietOCR: {exc}") from exc
        self.model_version = f"VietOCR {vietocr_version} ({variant})"
        self.configuration = {
            "preprocess": self.preprocess,
            "min_text_side": self.min_text_side,
            "variant": variant,
            "mix_chinese": language == "vi+zh-Hant",
            "detection_language": self.detection_language,
            "tesseract_config": TESSERACT_CONFIG,
        }

    def _load_page(self, image_path: Path) -> tuple[Image.Image, dict[str, object]]:
        if not self.preprocess:
            with Image.open(image_path) as image:
                return image.convert("L"), {"mode": "disabled"}
        prepared: PreparedPage = prepare_page(
            image_path,
            min_text_side=self.min_text_side,
        )
        return prepared.image, prepared.metadata()

    def _crop(self, image: Image.Image, line: _Line) -> Image.Image:
        x0, y0, x1, y1 = line.bbox
        return image.crop(
            (
                max(0, int(x0) - 8),
                max(0, int(y0) - 6),
                min(image.width, int(x1) + 8),
                min(image.height, int(y1) + 6),
            )
        ).convert("RGB")

    def _recognize_chinese(self, crop: Image.Image) -> str:
        return normalize_text(
            self._pytesseract.image_to_string(
                crop,
                lang="chi_tra",
                config=_CHINESE_CONFIG,
            )
        )

    def recognize(self, image_path: Path) -> tuple[PageOCRContent, object]:
        raw_response: object | None = None
        try:
            image, preprocess_metadata = self._load_page(image_path)
            width, height = image.size
            data = self._pytesseract.image_to_data(
                image,
                lang=self.detection_language,
                config=TESSERACT_CONFIG,
                output_type=self._pytesseract.Output.DICT,
            )
            if not isinstance(data, dict):
                raise Step4Error("Tesseract returned a non-object TSV result.")
            lines = _line_boxes(data)
            raw_lines: list[dict[str, object]] = []
            regions: list[LayoutRegion] = []
            paragraphs: list[list[TextLine]] = []
            previous_paragraph: tuple[int, int] | None = None

            for line in lines:
                crop = self._crop(image, line)
                text = ""
                source = ""
                confidence: float | None = None
                if self.language == "vi+zh-Hant":
                    chinese_text = self._recognize_chinese(crop)
                    if (
                        _cjk_count(chinese_text) >= 2
                        and _cjk_share(chinese_text) >= 0.3
                    ):
                        text = chinese_text
                        source = "tesseract-chi-tra"
                if not text:
                    try:
                        prediction, probability = self._predictor.predict(
                            crop,
                            return_prob=True,
                        )
                    except Exception as exc:
                        raise Step4Error(
                            f"VietOCR failed on line {line.key}: {exc}"
                        ) from exc
                    text = normalize_text(_merge_symbol_tokens(
                        str(prediction),
                        line.tesseract_text,
                    ))
                    confidence = float(probability)
                    source = "vietocr"
                if is_noise_text(text):
                    text = line.tesseract_text
                    source = "tesseract-fallback"
                if not text:
                    continue
                text_line = TextLine(
                    text=text,
                    bbox=line.bbox,
                )
                if line.paragraph != previous_paragraph:
                    paragraphs.append([])
                    previous_paragraph = line.paragraph
                paragraphs[-1].append(text_line)
                regions.append(
                    LayoutRegion(
                        type="text",
                        text=text,
                        bbox=normalize_box(line.bbox, width, height),
                    )
                )
                raw_lines.append(
                    {
                        "key": list(line.key),
                        "bbox": list(line.bbox),
                        "tesseract_text": line.tesseract_text,
                        "text": text,
                        "source": source,
                        "confidence": confidence,
                    }
                )

            raw_response = {
                "engine": self.model_name,
                "version": self.model_version,
                "language": self.language,
                "variant": self.variant,
                "detection": {
                    "engine": "Tesseract",
                    "version": self._tesseract_version,
                    "language": self.detection_language,
                    "config": TESSERACT_CONFIG,
                },
                "preprocess": preprocess_metadata,
                "lines": raw_lines,
                "tesseract_response": data,
            }
            return (
                PageOCRContent(
                    markdown=markdown_from_paragraphs(
                        paragraphs,
                        self.language,
                    ),
                    regions=regions,
                ),
                raw_response,
            )
        except Step4Error as exc:
            if getattr(exc, "raw_response", None) is not None:
                raise
            raise Step4Error(str(exc), raw_response) from exc
        except (OSError, UnidentifiedImageError) as exc:
            raise Step4Error(
                f"Cannot read OCR page image {image_path}: {exc}"
            ) from exc
        except Exception as exc:
            raise Step4Error(
                f"VietOCR inference failed for {image_path}: {exc}"
            ) from exc
