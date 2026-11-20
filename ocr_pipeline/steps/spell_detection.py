import importlib
import json
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


SUPPORTED_LANGUAGES = {"vi", "en", "fr", "zh-Hant"}
TRADITIONAL_MARKERS = set("體學習國語這個臺灣點書龍門後發現")
VIETNAMESE_MARKERS = set("ăâđêôơưĂÂĐÊÔƠƯ")
VIETNAMESE_STOPWORDS = {"và", "của", "là", "có", "được", "trong", "cho"}
ENGLISH_STOPWORDS = {"the", "and", "is", "are", "of", "to", "in", "with"}
FRENCH_STOPWORDS = {"le", "la", "les", "des", "une", "est", "et", "dans"}
LATIN_CHAR = r"[^\W\d_\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002FA1F]"
LATIN_TOKEN = re.compile(rf"{LATIN_CHAR}+(?:[’']{LATIN_CHAR}+)*", re.UNICODE)
HAN_TOKEN = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002FA1F]+"
)


class SpellDetectionError(Exception):
    """Raised when spelling inputs or language resources are invalid."""


@dataclass(frozen=True)
class LanguageGuess:
    language: str | None
    confidence: float
    reason: str


@dataclass(frozen=True)
class TextSpan:
    start: int
    end: int
    text: str
    language: str


def normalize_language(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    language = value.strip().replace("_", "-")
    aliases = {
        "vi-VN": "vi",
        "vie": "vi",
        "en-US": "en",
        "en-GB": "en",
        "eng": "en",
        "fr-FR": "fr",
        "fra": "fr",
        "fre": "fr",
        "zh-TW": "zh-Hant",
        "zh-HK": "zh-Hant",
        "zh-Hant-TW": "zh-Hant",
        "zh-Hant-HK": "zh-Hant",
    }
    if language in aliases:
        return aliases[language]
    if language == "vi+zh-Hant":
        return language
    if language in SUPPORTED_LANGUAGES:
        return language
    return None


def detect_language(text: str, hint: object = None) -> LanguageGuess:
    hinted = normalize_language(hint)
    if hinted is not None:
        return LanguageGuess(hinted, 1.0, "ocr-metadata-or-cli")

    normalized = unicodedata.normalize("NFC", text)
    words = {word.casefold() for word in LATIN_TOKEN.findall(normalized)}
    has_han = bool(HAN_TOKEN.search(normalized))
    has_vietnamese = any(character in VIETNAMESE_MARKERS for character in normalized)
    vi_hits = len(words & VIETNAMESE_STOPWORDS)
    en_hits = len(words & ENGLISH_STOPWORDS)
    fr_hits = len(words & FRENCH_STOPWORDS)
    has_traditional_marker = any(character in TRADITIONAL_MARKERS for character in normalized)

    if has_han and has_traditional_marker:
        language = "vi+zh-Hant" if has_vietnamese else "zh-Hant"
        return LanguageGuess(language, 0.98, "traditional-chinese-script-marker")
    if has_han:
        return LanguageGuess(None, 0.0, "chinese-script-ambiguous")
    if has_vietnamese or vi_hits >= 2:
        return LanguageGuess("vi", 0.98 if has_vietnamese else 0.85, "vietnamese-markers")
    if fr_hits >= 2 and fr_hits > en_hits:
        return LanguageGuess("fr", 0.85, "french-stopwords")
    if en_hits >= 2 and en_hits > fr_hits:
        return LanguageGuess("en", 0.85, "english-stopwords")
    return LanguageGuess(None, 0.0, "insufficient-language-evidence")


def _dictionary_language(value: object) -> str:
    language = normalize_language(value)
    if language == "vi+zh-Hant":
        raise SpellDetectionError(
            "Dictionary entries must use one language at a time."
        )
    if language is None:
        raise SpellDetectionError(f"Unsupported dictionary language: {value!r}")
    return language


def _normalize_word(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def load_dictionary(
    path: Path | None,
) -> tuple[
    dict[str, dict[str, int]],
    dict[str, set[str]],
    dict[str, set[str]],
]:
    if path is None:
        empty_terms = {language: set() for language in SUPPORTED_LANGUAGES}
        return {}, empty_terms, {language: set() for language in SUPPORTED_LANGUAGES}
    try:
        data: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SpellDetectionError(f"Cannot read spellcheck dictionary {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SpellDetectionError("Spellcheck dictionary must be a JSON object.")

    words: dict[str, dict[str, int]] = {}
    domain_terms = {language: set() for language in SUPPORTED_LANGUAGES}
    context_phrases = {language: set() for language in SUPPORTED_LANGUAGES}
    for raw_language, raw_section in data.items():
        language = _dictionary_language(raw_language)
        if not isinstance(raw_section, dict):
            raise SpellDetectionError(
                f"Dictionary section {raw_language!r} must be an object."
            )
        raw_words = raw_section.get("words", {})
        if isinstance(raw_words, list):
            word_items = [(word, 1) for word in raw_words]
        elif isinstance(raw_words, dict):
            word_items = list(raw_words.items())
        else:
            raise SpellDetectionError(
                f"Dictionary words for {raw_language!r} must be an array or object."
            )
        language_words: dict[str, int] = {}
        for raw_word, raw_frequency in word_items:
            if (
                not isinstance(raw_word, str)
                or not raw_word.strip()
                or type(raw_frequency) is not int
                or raw_frequency <= 0
            ):
                raise SpellDetectionError(
                    f"Invalid word or frequency in {raw_language!r} dictionary."
                )
            word = _normalize_word(raw_word)
            language_words[word] = raw_frequency
        words[language] = language_words

        raw_domain = raw_section.get("domain_terms", [])
        if not isinstance(raw_domain, list) or not all(
            isinstance(term, str) and term.strip() for term in raw_domain
        ):
            raise SpellDetectionError(
                f"Domain terms for {raw_language!r} must be non-empty strings."
            )
        domain_terms[language].update(
            _normalize_word(term)
            for term in raw_domain
        )

        raw_context = raw_section.get("context_phrases", [])
        if not isinstance(raw_context, list) or not all(
            isinstance(phrase, str) and phrase.strip()
            for phrase in raw_context
        ):
            raise SpellDetectionError(
                f"Context phrases for {raw_language!r} must be non-empty strings."
            )
        context_phrases[language].update(
            _normalize_word(phrase) for phrase in raw_context
        )
    return words, domain_terms, context_phrases


def _similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left.casefold(), right.casefold()).ratio()


class SpellDetector:
    def __init__(
        self,
        dictionary_words: dict[str, dict[str, int]],
        domain_terms: dict[str, set[str]],
        context_phrases: dict[str, set[str]],
    ) -> None:
        self.dictionary_words = dictionary_words
        self.domain_terms = domain_terms
        self.context_phrases = context_phrases
        self._spellcheckers: dict[str, Any] = {}
        self.warnings: list[str] = []

    def _spellchecker(self, language: str) -> Any | None:
        if language not in {"en", "fr"}:
            return None
        if language in self._spellcheckers:
            return self._spellcheckers[language]
        try:
            spellchecker_module = importlib.import_module("spellchecker")
        except ImportError:
            if not self.dictionary_words.get(language):
                raise SpellDetectionError(
                    "English/French checking requires pyspellchecker; install "
                    "the optional dependencies with `python -m pip install "
                    "-e \".[spellcheck]\"` or provide a local vocabulary."
                )
            self.warnings.append(
                f"{language}: pyspellchecker unavailable; using the supplied "
                "local vocabulary only."
            )
            self._spellcheckers[language] = None
            return None
        spellchecker_type = getattr(spellchecker_module, "SpellChecker")
        checker = spellchecker_type(language=language)
        self._spellcheckers[language] = checker
        return checker

    def is_known(self, word: str, language: str) -> bool:
        normalized = _normalize_word(word)
        if normalized in self.domain_terms.get(language, set()):
            return True
        if normalized in self.dictionary_words.get(language, {}):
            return True
        checker = self._spellchecker(language)
        return bool(checker is not None and checker.known([normalized]))

    def is_domain_term(self, word: str, language: str) -> bool:
        return (
            _normalize_word(word)
            in self.domain_terms.get(language, set())
        )

    def is_known_subspan(self, text: str, span: TextSpan) -> bool:
        normalized_text = unicodedata.normalize("NFC", text)
        known_terms = set(self.dictionary_words.get(span.language, {}))
        known_terms.update(self.domain_terms.get(span.language, set()))
        normalized_span = _normalize_word(span.text)
        for term in known_terms:
            if len(term) <= len(normalized_span):
                continue
            pattern = re.compile(
                r"(?<!\w)" + r"\s+".join(re.escape(part) for part in term.split())
                + r"(?!\w)",
                re.IGNORECASE,
            )
            if any(
                match.start() <= span.start
                and span.end <= match.end()
                for match in pattern.finditer(normalized_text)
            ):
                return True
        return False

    def candidates(self, word: str, language: str) -> dict[str, int]:
        normalized = _normalize_word(word)
        candidates: dict[str, int] = {}
        checker = self._spellchecker(language)
        if checker is not None:
            for candidate in checker.candidates(normalized) or ():
                candidate_word = unicodedata.normalize("NFC", candidate).casefold()
                if candidate_word != normalized:
                    candidates[candidate_word] = max(
                        1,
                        int(checker.word_frequency[candidate_word]),
                    )

        max_distance = (
            1
            if language == "zh-Hant" or len(normalized.replace(" ", "")) < 6
            else 2
        )
        source_length = len(normalized.replace(" ", ""))
        for candidate, frequency in self.dictionary_words.get(language, {}).items():
            candidate_length = len(candidate.replace(" ", ""))
            if candidate == normalized or abs(candidate_length - source_length) > max_distance:
                continue
            minimum_similarity = 0.45 if language == "zh-Hant" else 0.72
            if _similarity(normalized, candidate) >= minimum_similarity:
                candidates[candidate] = max(candidates.get(candidate, 0), frequency)
        for candidate in self.domain_terms.get(language, set()):
            candidate_length = len(candidate.replace(" ", ""))
            if candidate == normalized or abs(candidate_length - source_length) > max_distance:
                continue
            minimum_similarity = 0.45 if language == "zh-Hant" else 0.72
            if _similarity(normalized, candidate) >= minimum_similarity:
                candidates[candidate] = max(candidates.get(candidate, 0), 1)
        return candidates

    def frequency(self, word: str, language: str) -> int:
        normalized = _normalize_word(word)
        custom_frequency = self.dictionary_words.get(language, {}).get(normalized)
        if custom_frequency is not None:
            return custom_frequency
        checker = self._spellchecker(language)
        if checker is None:
            return 0
        return int(checker.word_frequency[normalized])


def text_spans(text: str, language: str) -> list[TextSpan]:
    spans: list[TextSpan] = []
    for match in LATIN_TOKEN.finditer(text):
        token_language = "vi" if language == "vi+zh-Hant" else language
        if token_language in {"vi", "en", "fr"}:
            spans.append(
                TextSpan(match.start(), match.end(), match.group(), token_language)
            )
    if language in {"zh-Hant", "vi+zh-Hant"}:
        for match in HAN_TOKEN.finditer(text):
            spans.extend(
                TextSpan(
                    match.start() + offset,
                    match.start() + offset + 1,
                    character,
                    "zh-Hant",
                )
                for offset, character in enumerate(match.group())
            )
    return sorted(spans, key=lambda span: (span.start, span.end))


def candidate_similarity(source: str, candidate: str) -> float:
    return _similarity(_normalize_word(source), _normalize_word(candidate))
