import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from ocr_pipeline.cli import main
from ocr_pipeline.steps.spell_detection import SpellDetector
from ocr_pipeline.steps.step4_1_spellcheck import spellcheck_document


class SpellDetectorTests(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("spellchecker"), "pyspellchecker is not installed"
    )
    def test_pyspellchecker_candidates_include_word_frequency(self) -> None:
        detector = SpellDetector({}, {}, {})

        candidates = detector.candidates("exampel", "en")

        self.assertGreater(candidates["example"], 0)
        self.assertGreater(detector.frequency("example", "en"), 0)
        self.assertEqual(detector.frequency("exampel", "en"), 0)


class Step41Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.batch = Path(self.temporary_directory.name) / "batch"
        self.source = self.batch / "work" / "step-4-ocr" / "B0012-01-01"
        self.pages = self.source / "pages"
        self.pages.mkdir(parents=True)
        (self.batch / "raw").mkdir(parents=True)
        self.document_id = "B0012-01-01"
        self.markdown = self.pages / "page-00001.md"
        self.page_json = self.pages / "page-00001.json"
        self.markdown.write_text("This is an exampel.", encoding="utf-8")
        self.page_json.write_text(
            json.dumps(
                {
                    "page_number": 1,
                    "status": "success",
                    "regions": [
                        {
                            "type": "text",
                            "text": "This is an exampel.",
                            "bbox": [0.1, 0.2, 0.8, 0.9],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        self._write_json(
            self.source / "ocr-report.json",
            {
                "schema_version": 1,
                "document_id": self.document_id,
                "model": "test-ocr",
                "model_version": "1",
                "language": "en",
                "status": "pass",
                "pages": [
                    {
                        "page_number": 1,
                        "status": "success",
                        "markdown_file": "pages/page-00001.md",
                    }
                ],
            },
        )
        self.dictionary = self.batch / "dictionary.json"
        self._write_json(
            self.dictionary,
            {
                "en": {
                    "words": {
                        "this": 100,
                        "is": 100,
                        "an": 100,
                        "example": 100,
                        "café": 100,
                    },
                    "domain_terms": ["Acme"],
                }
            },
        )
        self.output = (
            self.batch / "work" / "step-4.1-spellcheck" / self.document_id
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @staticmethod
    def _write_json(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def _run(self, threshold: float = 0.95) -> dict[str, object]:
        return spellcheck_document(
            self.batch,
            self.document_id,
            self.source,
            self.output,
            dictionary_file=self.dictionary,
            confidence_threshold=threshold,
        )

    @staticmethod
    def _decision_counts(report: dict[str, object]) -> dict[str, object]:
        decisions = report.get("decisions")
        if not isinstance(decisions, dict):
            raise AssertionError("Spellcheck report has invalid decision counts.")
        return decisions

    def test_auto_correction_is_logged_and_original_layout_is_immutable(self) -> None:
        original_markdown = self.markdown.read_bytes()
        original_page_json = self.page_json.read_bytes()

        report = self._run()

        corrected = (self.output / "pages" / "page-00001.md").read_text(
            encoding="utf-8"
        )
        self.assertEqual(corrected, "This is an example.")
        self.assertEqual(self.markdown.read_bytes(), original_markdown)
        self.assertEqual(self.page_json.read_bytes(), original_page_json)
        self.assertEqual(
            (self.output / "pages" / "page-00001.json").read_bytes(),
            original_page_json,
        )
        log_lines = (
            self.output / "correction-log.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        decision = json.loads(log_lines[0])
        self.assertEqual(decision["action"], "AUTO_CORRECT")
        self.assertEqual(decision["original"], "exampel")
        self.assertEqual(decision["replacement"], "example")
        self.assertTrue(decision["evidence"]["context_available"])
        self.assertEqual(self._decision_counts(report)["AUTO_CORRECT"], 1)

    def test_threshold_keeps_low_confidence_candidate(self) -> None:
        report = self._run(threshold=0.98)

        corrected = (self.output / "pages" / "page-00001.md").read_text(
            encoding="utf-8"
        )
        decision = json.loads(
            (self.output / "correction-log.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()[0]
        )
        self.assertEqual(corrected, "This is an exampel.")
        self.assertEqual(decision["action"], "KEEP_ORIGINAL")
        decisions = self._decision_counts(report)
        self.assertEqual(decisions["AUTO_CORRECT"], 0)
        self.assertEqual(decisions["KEEP_ORIGINAL"], 1)

    def test_nfc_analysis_preserves_unchanged_source_codepoints(self) -> None:
        original = "cafe\u0301 is an exampel."
        self.markdown.write_text(original, encoding="utf-8")

        self._run()

        self.assertEqual(
            (self.output / "pages" / "page-00001.md").read_text(encoding="utf-8"),
            "cafe\u0301 is an example.",
        )
        self.assertEqual(self.markdown.read_text(encoding="utf-8"), original)

    def test_context_phrase_resolves_competing_dictionary_candidates(self) -> None:
        self._write_json(
            self.dictionary,
            {
                "en": {
                    "words": {
                        "this": 100,
                        "is": 100,
                        "an": 100,
                        "example": 100,
                        "examples": 100,
                    },
                    "context_phrases": ["an example"],
                }
            },
        )

        self._run()

        self.assertEqual(
            (self.output / "pages" / "page-00001.md").read_text(encoding="utf-8"),
            "This is an example.",
        )
        decision = json.loads(
            (self.output / "correction-log.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()[0]
        )
        self.assertEqual(decision["action"], "AUTO_CORRECT")
        self.assertEqual(decision["replacement"], "example")

    def test_unresolved_language_keeps_text(self) -> None:
        report_path = self.source / "ocr-report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["language"] = None
        report_path.write_text(json.dumps(report), encoding="utf-8")

        self._run()

        decision = json.loads(
            (self.output / "correction-log.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()[0]
        )
        self.assertEqual(decision["action"], "KEEP_ORIGINAL")
        self.assertEqual(decision["reason"], "language-unresolved")
        self.assertEqual(
            (self.output / "pages" / "page-00001.md").read_text(encoding="utf-8"),
            "This is an exampel.",
        )

    def test_domain_terms_are_not_corrected(self) -> None:
        self.markdown.write_text("Acme", encoding="utf-8")

        report = self._run()

        self.assertEqual(self._decision_counts(report)["AUTO_CORRECT"], 0)
        self.assertEqual(
            (self.output / "pages" / "page-00001.md").read_text(encoding="utf-8"),
            "Acme",
        )

    def test_cli_tracks_step41_and_writes_separate_output(self) -> None:
        exit_code = main(
            [
                "spellcheck-document",
                "--batch",
                str(self.batch),
                "--document-id",
                self.document_id,
                "--dictionary-file",
                str(self.dictionary),
                "--run-id",
                "run-step41",
            ]
        )

        self.assertEqual(exit_code, 0)
        run_record = json.loads(
            (self.batch / "work" / "runs" / "run-step41.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(run_record["steps"]["step-4.1"]["status"], "pass")
        self.assertEqual(
            (self.output / "pages" / "page-00001.md").read_text(encoding="utf-8"),
            "This is an example.",
        )
        self.assertEqual(
            self.markdown.read_text(encoding="utf-8"),
            "This is an exampel.",
        )

    def test_vietnamese_candidate_can_correct_a_multi_syllable_phrase(self) -> None:
        self.markdown.write_text("thanh toan", encoding="utf-8")
        report_path = self.source / "ocr-report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["language"] = "vi"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        self._write_json(
            self.dictionary,
            {
                "vi": {
                    "words": {
                        "thanh": 100,
                        "toán": 100,
                        "thanh toán": 100,
                    }
                }
            },
        )

        report = self._run()

        self.assertEqual(
            (self.output / "pages" / "page-00001.md").read_text(encoding="utf-8"),
            "thanh toán",
        )
        self.assertEqual(self._decision_counts(report)["AUTO_CORRECT"], 1)

    def test_mixed_language_does_not_send_han_tokens_to_vietnamese_checker(self) -> None:
        self.markdown.write_text("thanh toan 學习", encoding="utf-8")
        report_path = self.source / "ocr-report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["language"] = "vi+zh-Hant"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        self._write_json(
            self.dictionary,
            {
                "vi": {
                    "words": {
                        "thanh": 100,
                        "toán": 100,
                        "thanh toán": 100,
                        "學習": 100,
                    }
                },
                "zh-Hant": {"words": {"學習": 100}},
            },
        )

        report = self._run()

        self.assertEqual(
            (self.output / "pages" / "page-00001.md").read_text(encoding="utf-8"),
            "thanh toán 學习",
        )
        decisions = self._decision_counts(report)
        self.assertEqual(decisions["AUTO_CORRECT"], 1)
        self.assertEqual(decisions["KEEP_ORIGINAL"], 1)

    def test_traditional_chinese_input_is_never_converted_to_dictionary_text(self) -> None:
        self.markdown.write_text("學习", encoding="utf-8")
        report_path = self.source / "ocr-report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["language"] = "zh-Hant"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        self._write_json(
            self.dictionary,
            {"zh-Hant": {"words": {"學習": 100}}},
        )

        self._run()

        corrected = (self.output / "pages" / "page-00001.md").read_text(
            encoding="utf-8"
        )
        self.assertEqual(corrected, "學习")
        log = (self.output / "correction-log.jsonl").read_text(encoding="utf-8")
        if log.strip():
            decision = json.loads(log.splitlines()[0])
            self.assertEqual(decision["action"], "KEEP_ORIGINAL")
            self.assertEqual(
                decision["reason"],
                "traditional-script-safety-not-proven",
            )
        self.markdown.write_text("学习", encoding="utf-8")

        self._run()

        self.assertEqual(
            (self.output / "pages" / "page-00001.md").read_text(encoding="utf-8"),
            "学习",
        )

    def test_known_traditional_phrase_is_not_expanded_from_a_substring(self) -> None:
        source_text = "乾。元亨利貞。"
        self.markdown.write_text(source_text, encoding="utf-8")
        report_path = self.source / "ocr-report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["language"] = "zh-Hant"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        self._write_json(
            self.dictionary,
            {"zh-Hant": {"words": {"元亨利貞": 1000}}},
        )

        self._run()

        self.assertEqual(
            (self.output / "pages" / "page-00001.md").read_text(encoding="utf-8"),
            source_text,
        )
        self.assertEqual(
            (self.output / "correction-log.jsonl")
            .read_text(encoding="utf-8"),
            "",
        )


if __name__ == "__main__":
    unittest.main()
