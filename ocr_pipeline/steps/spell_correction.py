import importlib
import re
import unicodedata
from dataclasses import asdict, dataclass, replace
from difflib import SequenceMatcher
from typing import Literal

from ocr_pipeline.steps.spell_detection import (
    LATIN_TOKEN,
    SpellDetector,
    TextSpan,
    candidate_similarity,
    text_spans,
)


Action = Literal["AUTO_CORRECT", "KEEP_ORIGINAL"]
ACTION_AUTO_CORRECT: Action = "AUTO_CORRECT"
ACTION_KEEP_ORIGINAL: Action = "KEEP_ORIGINAL"
PROTECTED_MARKDOWN = re.compile(
    r"```[\s\S]*?```|~~~[\s\S]*?~~~|`[^`\n]*`|"
    r"(?<=\]\()[^)\n]+(?=\))|https?://\S+",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CandidateScore:
    text: str
    confidence: float
    similarity: float
    candidate_frequency: int
    source_frequency: int
    context_score: float | None


@dataclass(frozen=True)
class SpellDecision:
    action: Action
    language: str | None
    start: int | None
    end: int | None
    original: str
    replacement: str | None
    confidence: float
    reason: str
    candidates: list[dict[str, object]]
    evidence: dict[str, object]


def _protected_ranges(text: str) -> list[tuple[int, int]]:
    return [(match.start(), match.end()) for match in PROTECTED_MARKDOWN.finditer(text)]


def _overlaps_protected(
    start: int,
    end: int,
    protected_ranges: list[tuple[int, int]],
) -> bool:
    return any(
        start < protected_end and end > protected_start
        for protected_start, protected_end in protected_ranges
    )


def _is_proper_name(text: str) -> bool:
    if text.isupper() and any(character.isalpha() for character in text):
        return True
    return text.istitle()


def _traditional_script_guard(source: str, candidate: str) -> bool:
    try:
        opencc_module = importlib.import_module("opencc")
        opencc_type = getattr(opencc_module, "OpenCC")
    except (AttributeError, ImportError):
        return False
    converter = opencc_type("s2t")
    return (
        converter.convert(candidate) == candidate
        and converter.convert(source) != candidate
    )


def traditional_script_guard_available() -> bool:
    try:
        opencc_module = importlib.import_module("opencc")
    except ImportError:
        return False
    return callable(getattr(opencc_module, "OpenCC", None))


def _candidate_scores(
    source: str,
    language: str,
    detector: SpellDetector,
    text: str,
    span: TextSpan,
) -> list[CandidateScore]:
    candidate_map = detector.candidates(source, language)
    ranked: list[CandidateScore] = []
    source_frequency = detector.frequency(source, language)
    for candidate, candidate_frequency in candidate_map.items():
        similarity = candidate_similarity(source, candidate)
        context_score = _context_score(text, span, detector, candidate)
        frequency_gain = (
            candidate_frequency / (candidate_frequency + source_frequency)
            if candidate_frequency + source_frequency
            else 0.0
        )
        if context_score is None:
            confidence = (0.45 + 0.3 * similarity + 0.15 * frequency_gain) / 0.9
        else:
            confidence = (
                0.45
                + 0.3 * similarity
                + 0.15 * frequency_gain
                + 0.1 * context_score
            )
        confidence = min(1.0, confidence)
        ranked.append(
            CandidateScore(
                text=candidate,
                confidence=round(confidence, 4),
                similarity=round(similarity, 4),
                candidate_frequency=candidate_frequency,
                source_frequency=source_frequency,
                context_score=context_score,
            )
        )
    return sorted(
        ranked,
        key=lambda item: (
            item.confidence,
            item.similarity,
            item.text,
        ),
        reverse=True,
    )


def _context_score(
    text: str,
    span: TextSpan,
    detector: SpellDetector,
    candidate: str,
) -> float | None:
    if span.language not in {"vi", "en", "fr"}:
        return None
    before = text[: span.start]
    after = text[span.end :]
    previous_matches = list(LATIN_TOKEN.finditer(before))
    following_match = LATIN_TOKEN.search(after)
    neighboring: list[str] = []
    if previous_matches:
        previous = previous_matches[-1]
        if not re.search(r"[.!?]\s*$", before[previous.end() :]):
            neighboring.append(previous.group())
    if following_match is not None and not re.match(
        r"\s*[.!?]", after[: following_match.start()]
    ):
        neighboring.append(following_match.group())
    if not neighboring:
        return None
    context_phrases = detector.context_phrases.get(span.language, set())
    if context_phrases:
        normalized_candidate = " ".join(
            unicodedata.normalize("NFC", candidate).casefold().split()
        )
        normalized_neighbors = [
            " ".join(unicodedata.normalize("NFC", word).casefold().split())
            for word in neighboring
        ]
        contexts: set[str] = set()
        if normalized_neighbors:
            contexts.add(
                f"{normalized_neighbors[0]} {normalized_candidate}"
            )
            contexts.add(
                f"{normalized_candidate} {normalized_neighbors[-1]}"
            )
        if len(normalized_neighbors) == 2:
            contexts.add(
                f"{normalized_neighbors[0]} "
                f"{normalized_candidate} "
                f"{normalized_neighbors[1]}"
            )
        return float(bool(contexts & context_phrases))
    return sum(
        detector.is_known(word, span.language) for word in neighboring
    ) / len(neighboring)


def _preserve_surface_form(source: str, candidate: str) -> str:
    source_words = re.findall(r"\S+", source)
    candidate_words = candidate.split()
    separators = re.findall(r"\s+", source)
    if len(source_words) == len(candidate_words):
        pieces: list[str] = []
        for index, word in enumerate(candidate_words):
            source_word = source_words[index]
            if source_word.isupper():
                word = word.upper()
            elif source_word.istitle():
                word = word.title()
            pieces.append(word)
            if index < len(separators):
                pieces.append(separators[index])
        return "".join(pieces)
    if source.isupper():
        return candidate.upper()
    if source.istitle():
        return candidate.title()
    return candidate


def _decision_for_span(
    text: str,
    span: TextSpan,
    detector: SpellDetector,
    threshold: float,
) -> SpellDecision | None:
    source = span.text
    if detector.is_known(source, span.language) or detector.is_known_subspan(
        text,
        span,
    ):
        return None
    candidates = _candidate_scores(
        source,
        span.language,
        detector,
        text,
        span,
    )
    if not candidates:
        return None
    best = candidates[0]
    best_text = best.text
    best_confidence = best.confidence
    evidence: dict[str, object] = {
        "dictionary_match": True,
        "candidate_similarity": best.similarity,
        "frequency_gain": (
            best.candidate_frequency
            / (best.candidate_frequency + best.source_frequency)
            if best.candidate_frequency + best.source_frequency
            else 0.0
        ),
        "context_score": best.context_score,
        "context_available": best.context_score is not None,
        "ocr_confidence_available": False,
        "visual_evidence_used": False,
    }

    reason = "confidence_below_threshold"
    auto_correct = best_confidence >= threshold
    if (
        len(candidates) > 1
        and best_confidence - candidates[1].confidence < 0.08
    ):
        auto_correct = False
        reason = "ambiguous_candidates"
    elif _is_proper_name(source):
        auto_correct = False
        reason = "possible_proper_name"
    elif span.language == "zh-Hant" and not _traditional_script_guard(
        source, best_text
    ):
        auto_correct = False
        reason = "traditional-script-safety-not-proven"
    elif auto_correct:
        reason = "unique_candidate_meets_heuristic_threshold"

    return SpellDecision(
        action=ACTION_AUTO_CORRECT if auto_correct else ACTION_KEEP_ORIGINAL,
        language=span.language,
        start=span.start,
        end=span.end,
        original=source,
        replacement=(
            _preserve_surface_form(source, best_text) if auto_correct else None
        ),
        confidence=round(best_confidence, 4),
        reason=reason,
        candidates=[asdict(candidate) for candidate in candidates[:5]],
        evidence=evidence,
    )


def _latin_spans(
    text: str,
    language: str,
    protected_ranges: list[tuple[int, int]],
) -> list[TextSpan]:
    spans = text_spans(text, language)
    return [
        span
        for span in spans
        if span.language in {"vi", "en", "fr"}
        and not _overlaps_protected(span.start, span.end, protected_ranges)
    ]


def _candidate_windows(
    text: str,
    spans: list[TextSpan],
    language: str,
) -> list[TextSpan]:
    if language not in {"vi", "zh-Hant"}:
        return spans
    windows: list[TextSpan] = []
    max_words = 3 if language == "vi" else 6
    for start_index, first in enumerate(spans):
        for count in range(1, min(max_words, len(spans) - start_index) + 1):
            selected = spans[start_index : start_index + count]
            if any(span.language != first.language for span in selected):
                break
            if any(
                spans[index].end != spans[index + 1].start
                and language == "zh-Hant"
                for index in range(start_index, start_index + count - 1)
            ):
                break
            source = text[first.start : selected[-1].end]
            if len(source) < 2:
                continue
            windows.append(
                TextSpan(
                    first.start,
                    selected[-1].end,
                    source,
                    first.language,
                )
            )
    return windows


def _decide_normalized_text(
    text: str,
    language: str | None,
    detector: SpellDetector,
    threshold: float = 0.95,
) -> list[SpellDecision]:
    if language not in {"vi", "en", "fr", "zh-Hant", "vi+zh-Hant"}:
        return [
            SpellDecision(
                action=ACTION_KEEP_ORIGINAL,
                language=None,
                start=None,
                end=None,
                original="",
                replacement=None,
                confidence=0.0,
                reason="language-unresolved",
                candidates=[],
                evidence={
                    "context_available": False,
                    "ocr_confidence_available": False,
                    "visual_evidence_used": False,
                },
            )
        ]

    protected_ranges = _protected_ranges(text)
    spans: list[TextSpan] = []
    if language in {"vi", "en", "fr", "vi+zh-Hant"}:
        spans.extend(_latin_spans(text, language, protected_ranges))
    if language in {"zh-Hant", "vi+zh-Hant"}:
        spans.extend(
            span
            for span in text_spans(text, language)
            if span.language == "zh-Hant"
            and not _overlaps_protected(span.start, span.end, protected_ranges)
        )

    proposals: list[SpellDecision] = []
    for span in spans:
        if span.language == "zh-Hant":
            # Chinese dictionary terms are matched on bounded substrings so
            # tokenization never converts or normalizes script variants.
            continue
        decision = _decision_for_span(text, span, detector, threshold)
        if decision is not None:
            proposals.append(decision)

    for selected_language in ("vi", "zh-Hant"):
        language_spans = [span for span in spans if span.language == selected_language]
        for span in _candidate_windows(text, language_spans, selected_language):
            if selected_language == "zh-Hant" and len(span.text) > 6:
                continue
            decision = _decision_for_span(text, span, detector, threshold)
            if decision is not None:
                proposals.append(decision)

    selected_decisions: list[SpellDecision] = []
    for decision in sorted(
        proposals,
        key=lambda item: (
            item.start if item.start is not None else -1,
            -(item.end or 0) + (item.start or 0),
        ),
    ):
        if decision.start is None or decision.end is None:
            continue
        if selected_decisions and decision.start < (selected_decisions[-1].end or 0):
            continue
        selected_decisions.append(decision)

    return selected_decisions


def _normalization_offsets(
    original: str,
    normalized: str,
) -> tuple[list[int], list[int]]:
    if original == normalized:
        return list(range(len(original))), list(range(1, len(original) + 1))
    starts = [0] * len(normalized)
    ends = [0] * len(normalized)
    matcher = SequenceMatcher(None, original, normalized, autojunk=False)
    for tag, original_start, original_end, normalized_start, normalized_end in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(normalized_end - normalized_start):
                starts[normalized_start + offset] = original_start + offset
                ends[normalized_start + offset] = original_start + offset + 1
        elif tag in {"replace", "insert"}:
            mapped_start = original_start
            mapped_end = original_end if tag == "replace" else original_start
            for offset in range(normalized_start, normalized_end):
                starts[offset] = mapped_start
                ends[offset] = mapped_end
    return starts, ends


def correct_text(
    text: str,
    language: str | None,
    detector: SpellDetector,
    threshold: float = 0.95,
) -> tuple[str, list[SpellDecision]]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Spellcheck confidence threshold must be between 0 and 1.")
    normalized = unicodedata.normalize("NFC", text)
    decisions = _decide_normalized_text(
        normalized,
        language,
        detector,
        threshold,
    )
    starts, ends = _normalization_offsets(text, normalized)
    mapped_decisions: list[SpellDecision] = []
    for decision in decisions:
        if decision.start is None or decision.end is None or decision.end <= decision.start:
            mapped_decisions.append(decision)
            continue
        source_start = starts[decision.start]
        source_end = ends[decision.end - 1]
        mapped_decisions.append(
            replace(
                decision,
                start=source_start,
                end=source_end,
                original=text[source_start:source_end],
            )
        )

    corrected = text
    for decision in reversed(mapped_decisions):
        if decision.action == ACTION_AUTO_CORRECT and decision.replacement is not None:
            assert decision.start is not None and decision.end is not None
            corrected = (
                corrected[: decision.start]
                + decision.replacement
                + corrected[decision.end :]
            )
    return corrected, mapped_decisions


def decision_dict(decision: SpellDecision) -> dict[str, object]:
    return asdict(decision)
