import json
import tempfile
import unittest
from pathlib import Path

from ocr_pipeline.steps.run_tracking import record_step_run


class RunTrackingTests(unittest.TestCase):
    def test_changed_spellcheck_output_marks_assembly_and_quality_stale(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            batch = Path(temporary_directory) / "batch"
            source = batch / "work" / "step-4-ocr" / "source.md"
            spellcheck = batch / "work" / "step-4.1-spellcheck" / "corrected.md"
            assembly = batch / "work" / "step-6-assemble" / "document.md"
            quality = batch / "work" / "step-8-quality" / "report.json"
            for path in (source, spellcheck, assembly, quality):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("first version", encoding="utf-8")

            record_step_run(
                batch, "run-test", "step-4.1", "pass", [source], [spellcheck]
            )
            record_step_run(
                batch, "run-test", "step-6", "pass", [spellcheck], [assembly]
            )
            record_step_run(
                batch, "run-test", "step-8", "pass", [assembly], [quality]
            )
            spellcheck.write_text("updated correction", encoding="utf-8")
            record_step_run(
                batch, "run-test", "step-4.1", "pass", [source], [spellcheck]
            )

            run_record = json.loads(
                (batch / "work" / "runs" / "run-test.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(run_record["steps"]["step-6"]["status"], "stale")
            self.assertEqual(run_record["steps"]["step-8"]["status"], "stale")

    def test_changed_input_marks_downstream_steps_stale(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            batch = Path(temporary_directory) / "batch"
            source = batch / "raw" / "scan.tif"
            step1_output = batch / "work" / "step-1" / "report.json"
            step2_output = batch / "work" / "step-2" / "report.json"
            source.parent.mkdir(parents=True)
            step1_output.parent.mkdir(parents=True)
            step2_output.parent.mkdir(parents=True)
            source.write_bytes(b"first scan")
            step1_output.write_text("first validation", encoding="utf-8")
            step2_output.write_text("first split", encoding="utf-8")

            record_step_run(batch, "run-test", "step-1", "pass", [source], [step1_output])
            record_step_run(batch, "run-test", "step-2", "pass", [step1_output], [step2_output])
            source.write_bytes(b"changed scan")
            record_step_run(batch, "run-test", "step-1", "pass", [source], [step1_output])

            run_record = json.loads(
                (batch / "work" / "runs" / "run-test.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(run_record["steps"]["step-1"]["status"], "pass")
            self.assertEqual(run_record["steps"]["step-2"]["status"], "stale")
            self.assertIn("step-1", run_record["steps"]["step-2"]["stale_reason"])


if __name__ == "__main__":
    unittest.main()
