import json
import tempfile
import unittest
from pathlib import Path

from ocr_pipeline.self_test import SelfTestError, run_self_test


class GeneratedPipelineTests(unittest.TestCase):
    def test_self_test_runs_all_steps_on_generated_scan_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "generated"

            result = run_self_test(output, "run-test")

            self.assertEqual(result.status, "pass")
            self.assertEqual(len(result.checks), 10)
            self.assertTrue((result.batch_root / "raw" / "B0012" / "01").is_dir())
            self.assertTrue((result.published_bundle / "document.md").is_file())
            report = json.loads(
                (output / "self-test-report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["status"], "pass")
            run_record = json.loads(
                (
                    result.batch_root
                    / "work"
                    / "runs"
                    / "run-test.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                {f"step-{step}" for step in range(1, 10)} | {"step-4.1"},
                set(run_record["steps"]),
            )

    def test_self_test_refuses_to_overwrite_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "existing"
            output.mkdir()

            with self.assertRaisesRegex(SelfTestError, "already exists"):
                run_self_test(output, "run-test")


if __name__ == "__main__":
    unittest.main()
