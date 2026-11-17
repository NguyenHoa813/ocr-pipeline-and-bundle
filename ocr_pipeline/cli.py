import argparse
import sys
from pathlib import Path

from ocr_pipeline.steps.step1_check_batch import (
    BatchCheckError,
    check_batch,
    resolve_scan_folder,
    write_reports,
)
from ocr_pipeline.steps.batch_manifest import (
    ManifestError,
    validate_batch_manifest,
    write_manifest_report,
)
from ocr_pipeline.steps.step2_split_documents import (
    Step2Error,
    ZXingBarcodeDetector,
    split_checked_batch,
    write_split_report,
)
from ocr_pipeline.steps.step3_flag_blank_pages import (
    Step3Error,
    flag_document_blank_pages,
    write_blank_page_report,
)
from ocr_pipeline.steps.step4_ocr_pages import (
    Step4Error,
    load_ocr_adapter,
    run_document_ocr,
    write_document_ocr,
)
from ocr_pipeline.steps.step5_cut_out_images import (
    Step5Error,
    cut_out_document_images,
    write_cut_out_report,
)
from ocr_pipeline.steps.step6_assemble_document import (
    Step6Error,
    assemble_document,
)
from ocr_pipeline.steps.step7_describe_images import (
    GeminiVertexAIAdapter,
    Step7Error,
    describe_document_images,
    write_image_description_report,
)
from ocr_pipeline.steps.step8_quality_check import (
    Step8Error,
    check_document_quality,
    write_quality_report,
)
from ocr_pipeline.steps.step9_local_storage import (
    Step9Error,
    publish_document_bundle,
)
from ocr_pipeline.steps.run_tracking import (
    RunTrackingError,
    new_run_id,
    record_step_run,
)
from ocr_pipeline.self_test import SelfTestError, run_self_test


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ocr-pipeline",
        description="Run local OCR pipeline steps.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser(
        "check-batch",
        help="Validate TIFF filenames, page sequences, page count, and box codes.",
    )
    check.add_argument(
        "--batch",
        type=Path,
        required=True,
        help="Batch root containing raw\\, box_list.csv, and optionally config.json.",
    )
    check.add_argument(
        "--box-list",
        type=Path,
        help="Box-list CSV; defaults to <batch>\\box_list.csv.",
    )
    check.add_argument(
        "--kind",
        choices=("folder", "book"),
        help="Unit type when no manifest is supplied (or to enforce one type).",
    )
    check.add_argument(
        "--output",
        type=Path,
        help="Report directory; defaults to <batch>\\work\\step-1-checks.",
    )
    check.add_argument(
        "--manifest",
        type=Path,
        help="Batch manifest; defaults to <batch>\\batch_manifest.yaml when present.",
    )

    split = commands.add_parser(
        "split-documents",
        help="Split checked folders at barcode separator sheets.",
    )
    split.add_argument(
        "--batch",
        type=Path,
        required=True,
        help="Batch root containing raw\\ and step-1 output under work\\.",
    )
    split.add_argument(
        "--check-report",
        type=Path,
        help="Step 1 batch-report.json; defaults to <batch>\\work\\step-1-checks.",
    )
    split.add_argument(
        "--output",
        type=Path,
        help="Report directory; defaults to <batch>\\work\\step-2-documents.",
    )

    blank = commands.add_parser(
        "flag-blank-pages",
        help="Measure dark-pixel share for pages in one split document.",
    )
    blank.add_argument(
        "--batch",
        type=Path,
        required=True,
        help="Batch root containing raw\\ and work\\ pipeline outputs.",
    )
    blank.add_argument(
        "--document-id",
        required=True,
        help="Document ID from Step 2, for example B0012-01-02.",
    )
    blank.add_argument(
        "--dark-pixel-share-threshold",
        type=float,
        required=True,
        help="Pilot-calibrated blank threshold, between 0 and 1.",
    )
    blank.add_argument(
        "--split-report",
        type=Path,
        help="Step 2 report; defaults to <batch>\\work\\step-2-documents.",
    )
    blank.add_argument(
        "--check-report",
        type=Path,
        help="Step 1 report; defaults to <batch>\\work\\step-1-checks.",
    )
    blank.add_argument(
        "--output",
        type=Path,
        help="Output JSON; defaults to <batch>\\work\\step-3-blank-pages"
        "\\<document-id>.json.",
    )

    ocr = commands.add_parser(
        "ocr-document",
        help="OCR each non-blank page using a built-in or installed OCR plugin.",
    )
    ocr.add_argument("--batch", type=Path, required=True)
    ocr.add_argument("--document-id", required=True)
    ocr.add_argument(
        "--model",
        default="paddle",
        help="OCR plugin name (default: paddle).",
    )
    ocr.add_argument(
        "--language",
        default="vi",
        help="Language code supported by the selected OCR adapter (vi, zh-Hant, fr, en).",
    )
    ocr.add_argument("--split-report", type=Path)
    ocr.add_argument("--check-report", type=Path)
    ocr.add_argument("--blank-report", type=Path)
    ocr.add_argument(
        "--output",
        type=Path,
        help="Output folder; defaults to <batch>\\work\\step-4-ocr"
        "\\<document-id>.",
    )

    cut = commands.add_parser(
        "cut-out-images",
        help="Crop figure regions from original TIFF pages.",
    )
    cut.add_argument(
        "--batch",
        type=Path,
        required=True,
        help="Batch root containing raw\\ and work\\ pipeline outputs.",
    )
    cut.add_argument(
        "--document-id",
        required=True,
        help="Document ID from Step 2, for example B0012-01-02.",
    )
    cut.add_argument(
        "--ocr-output",
        type=Path,
        help="Step 4 output folder; defaults to <batch>\\work\\step-4-ocr"
        "\\<document-id>.",
    )
    cut.add_argument(
        "--output",
        type=Path,
        help="Output folder; defaults to <batch>\\work\\step-5-images"
        "\\<document-id>.",
    )
    cut.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality for figure regions marked photo (1-100; default 95).",
    )

    assemble = commands.add_parser(
        "assemble-document",
        help="Assemble Step 1-5 and optional Step 7 outputs into a final bundle.",
    )
    assemble.add_argument(
        "--batch",
        type=Path,
        required=True,
        help="Batch root containing raw\\ and work\\ pipeline outputs.",
    )
    assemble.add_argument(
        "--document-id",
        required=True,
        help="Document ID from Step 2, for example B0012-01-02.",
    )
    assemble.add_argument(
        "--split-report",
        type=Path,
        help="Step 2 report; defaults to <batch>\\work\\step-2-documents.",
    )
    assemble.add_argument(
        "--check-report",
        type=Path,
        help="Step 1 report; defaults to <batch>\\work\\step-1-checks.",
    )
    assemble.add_argument(
        "--ocr-output",
        type=Path,
        help="Step 4 output folder; defaults to <batch>\\work\\step-4-ocr"
        "\\<document-id>.",
    )
    assemble.add_argument(
        "--image-output",
        type=Path,
        help="Step 5 output folder; defaults to <batch>\\work\\step-5-images"
        "\\<document-id>.",
    )
    assemble.add_argument(
        "--output",
        type=Path,
        help="Bundle folder; defaults to <batch>\\work\\step-6-assemble"
        "\\<document-id>.",
    )
    assemble.add_argument(
        "--description-report",
        type=Path,
        help="Step 7 JSON; defaults to <batch>\\work\\step-7-image-descriptions"
        "\\<document-id>.json when present.",
    )
    assemble.add_argument(
        "--skip-step7",
        action="store_true",
        help="Assemble figures using OCR labels without Step 7 descriptions.",
    )

    describe = commands.add_parser(
        "describe-images",
        help="Describe only Step 5 image crops with Gemini on Vertex AI.",
    )
    describe.add_argument("--batch", type=Path, required=True)
    describe.add_argument("--document-id", required=True)
    describe.add_argument("--project", required=True, help="Google Cloud project ID.")
    describe.add_argument("--location", required=True, help="Vertex AI location.")
    describe.add_argument("--model", default="gemini-2.5-flash")
    describe.add_argument(
        "--allow-cropped-image-upload",
        action="store_true",
        help="Explicitly confirm Step 5 crops may be sent to Vertex AI.",
    )
    describe.add_argument("--image-folder", type=Path)
    describe.add_argument("--cut-out-report", type=Path)
    describe.add_argument("--output", type=Path)

    quality = commands.add_parser(
        "check-bundle",
        help="Validate bundle structure and report independent OCR page flags.",
    )
    quality.add_argument("--batch", type=Path, required=True)
    quality.add_argument("--document-id", required=True)
    quality.add_argument("--bundle", type=Path)
    quality.add_argument("--ocr-report", type=Path)
    quality.add_argument("--vocabulary-file", type=Path)
    quality.add_argument(
        "--vocabulary-threshold",
        type=float,
        help="Pilot-calibrated minimum vocabulary ratio; requires vocabulary file.",
    )
    quality.add_argument("--output", type=Path)
    quality.add_argument("--flag-log", type=Path)

    publish = commands.add_parser(
        "publish-bundle",
        help="Atomically publish valid bundles or quarantine rejected bundles.",
    )
    publish.add_argument("--batch", type=Path, required=True)
    publish.add_argument("--document-id", required=True)
    publish.add_argument("--bundle", type=Path)
    publish.add_argument("--quality-report", type=Path)
    publish.add_argument("--storage-root", type=Path)

    self_test = commands.add_parser(
        "self-test",
        help="Generate synthetic scans and run all nine steps without cloud access.",
    )
    self_test.add_argument(
        "--output",
        type=Path,
        help="New output directory; defaults to a unique folder in the current directory.",
    )

    for subparser in commands.choices.values():
        subparser.add_argument(
            "--run-id",
            help="Pipeline run ID; auto-generated unless supplied to correlate steps.",
        )

    return parser


def _record_failed_run(args: argparse.Namespace, run_id: str, step: str) -> None:
    batch_root = args.batch.resolve()
    scan_folder = resolve_scan_folder(batch_root)
    default_outputs = {
        "step-1": batch_root / "work" / "step-1-checks",
        "step-2": batch_root / "work" / "step-2-documents",
        "step-3": batch_root / "work" / "step-3-blank-pages",
        "step-4": batch_root / "work" / "step-4-ocr",
        "step-5": batch_root / "work" / "step-5-images",
        "step-6": batch_root / "work" / "step-6-assemble",
        "step-7": batch_root / "work" / "step-7-image-descriptions",
        "step-8": batch_root / "work" / "step-8-quality",
        "step-9": batch_root / "storage" / "output",
    }
    inputs = [scan_folder]
    if step == "step-1":
        inputs.append(args.box_list or batch_root / "box_list.csv")
        manifest_path = args.manifest or batch_root / "batch_manifest.yaml"
        if manifest_path.exists():
            inputs.append(manifest_path)
    try:
        record_step_run(
            batch_root,
            run_id,
            step,
            "fail",
            inputs,
            [default_outputs[step]],
        )
    except RunTrackingError as exc:
        print(f"ERROR: Could not record failed run: {exc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_id = args.run_id or new_run_id()
    if args.command == "check-batch":
        try:
            batch_root = args.batch.resolve()
            scan_folder = resolve_scan_folder(batch_root)
            box_list = args.box_list or batch_root / "box_list.csv"
            output_folder = (
                args.output or batch_root / "work" / "step-1-checks"
            ).resolve()
            if output_folder.is_relative_to(scan_folder):
                raise BatchCheckError(
                    "Report output must be outside the raw scan folder."
                )
            manifest_path = args.manifest or batch_root / "batch_manifest.yaml"
            manifest_path = manifest_path.resolve()
            if not manifest_path.exists() and args.manifest is None:
                manifest_path = None
            if args.kind is None and manifest_path is None:
                raise BatchCheckError("Provide --kind or --manifest.")
            result = check_batch(scan_folder, box_list, args.kind, manifest_path)
            write_reports(result, output_folder)
            if manifest_path is not None:
                manifest_report, _ = validate_batch_manifest(
                    manifest_path,
                    expected_kind=args.kind,
                    unit_page_counts={
                        (unit.box_code, unit.unit_number): unit.page_count
                        for unit in result.units
                    },
                )
                write_manifest_report(
                    manifest_report,
                    output_folder / "batch-manifest-report.json",
                )
        except (BatchCheckError, OSError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            _record_failed_run(args, run_id, "step-1")
            return 2
        except ManifestError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            _record_failed_run(args, run_id, "step-1")
            return 2

        record_step_run(
            batch_root,
            run_id,
            "step-1",
            "pass" if result.status == "pass" else "fail",
            [scan_folder, box_list] + ([manifest_path] if manifest_path else []),
            [output_folder],
            {"kind": args.kind},
        )

        print(
            f"Checked {len(result.units)} unit(s): "
            f"{result.passed_count} passed, {result.failed_count} failed."
        )
        print(f"Reports: {output_folder}")
        return 0 if result.status == "pass" else 1

    if args.command == "split-documents":
        try:
            batch_root = args.batch.resolve()
            scan_folder = resolve_scan_folder(batch_root)
            check_report = args.check_report or (
                batch_root / "work" / "step-1-checks" / "batch-report.json"
            )
            output_folder = (
                args.output or batch_root / "work" / "step-2-documents"
            ).resolve()
            if output_folder.is_relative_to(scan_folder):
                raise Step2Error("Report output must be outside the raw scan folder.")
            result = split_checked_batch(
                batch_root,
                check_report,
                ZXingBarcodeDetector(),
            )
            write_split_report(result, output_folder)
        except (Step2Error, OSError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            _record_failed_run(args, run_id, "step-2")
            return 2

        record_step_run(
            batch_root,
            run_id,
            "step-2",
            "pass" if result.status == "pass" else "fail",
            [scan_folder, check_report],
            [output_folder],
        )
        passed = sum(unit.status == "pass" for unit in result.units)
        skipped = sum(unit.status == "skipped" for unit in result.units)
        failed = sum(unit.status == "fail" for unit in result.units)
        print(
            f"Split {passed} unit(s), skipped {skipped}, failed {failed}."
        )
        print(f"Reports: {output_folder}")
        return 0 if result.status == "pass" else 1

    if args.command == "flag-blank-pages":
        try:
            batch_root = args.batch.resolve()
            scan_folder = resolve_scan_folder(batch_root)
            output_path = (
                args.output
                or batch_root
                / "work"
                / "step-3-blank-pages"
                / f"{args.document_id}.json"
            ).resolve()
            if output_path.is_relative_to(scan_folder):
                raise Step3Error("Blank-page report must be outside raw scans.")
            result = flag_document_blank_pages(
                batch_root,
                args.document_id,
                args.dark_pixel_share_threshold,
                args.split_report,
                args.check_report,
            )
            write_blank_page_report(result, output_path)
        except (Step3Error, OSError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            _record_failed_run(args, run_id, "step-3")
            return 2

        record_step_run(
            batch_root,
            run_id,
            "step-3",
            "pass" if result.status == "pass" else "fail",
            [
                batch_root / "work" / "step-2-documents" / "split-report.json",
                batch_root / "work" / "step-1-checks" / "batch-report.json",
            ],
            [output_path],
            {"dark_pixel_share_threshold": args.dark_pixel_share_threshold},
        )
        blank_count = sum(page.is_blank for page in result.pages)
        print(
            f"Flagged {blank_count} blank page(s) out of {len(result.pages)} "
            f"for {result.document_id}."
        )
        print(f"Report: {output_path}")
        return 0

    if args.command == "ocr-document":
        try:
            batch_root = args.batch.resolve()
            scan_folder = resolve_scan_folder(batch_root)
            output_folder = (
                args.output
                or batch_root
                / "work"
                / "step-4-ocr"
                / args.document_id
            ).resolve()
            if output_folder.is_relative_to(scan_folder):
                raise Step4Error("OCR output must be outside raw scans.")
            adapter = load_ocr_adapter(args.model, language=args.language)
            report, normalized, raw = run_document_ocr(
                batch_root,
                args.document_id,
                adapter,
                args.split_report,
                args.check_report,
                args.blank_report,
            )
            write_document_ocr(report, normalized, raw, output_folder)
        except (Step4Error, OSError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            _record_failed_run(args, run_id, "step-4")
            return 2

        record_step_run(
            batch_root,
            run_id,
            "step-4",
            report.status,
            [
                scan_folder,
                args.split_report
                or batch_root / "work" / "step-2-documents" / "split-report.json",
                args.check_report
                or batch_root / "work" / "step-1-checks" / "batch-report.json",
                args.blank_report
                or batch_root
                / "work"
                / "step-3-blank-pages"
                / f"{args.document_id}.json",
            ],
            [output_folder],
            {"model": args.model, "language": args.language},
        )
        print(
            f"OCR {report.status} for {args.document_id} with {report.model}; "
            f"{len(report.pages)} page(s)."
        )
        print(f"Output: {output_folder}")
        return 0 if report.status == "pass" else 1

    if args.command == "cut-out-images":
        try:
            batch_root = args.batch.resolve()
            scan_folder = resolve_scan_folder(batch_root)
            output_folder = (
                args.output
                or batch_root
                / "work"
                / "step-5-images"
                / args.document_id
            ).resolve()
            if output_folder.is_relative_to(scan_folder):
                raise Step5Error("Image output must be outside raw scans.")
            report = cut_out_document_images(
                batch_root,
                args.document_id,
                args.ocr_output,
                output_folder,
                args.jpeg_quality,
            )
            write_cut_out_report(report, output_folder)
        except (Step5Error, OSError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            _record_failed_run(args, run_id, "step-5")
            return 2

        record_step_run(
            batch_root,
            run_id,
            "step-5",
            "pass" if report.status == "pass" else "fail",
            [
                args.ocr_output
                or batch_root / "work" / "step-4-ocr" / args.document_id
            ],
            [output_folder],
            {"jpeg_quality": args.jpeg_quality},
        )
        print(
            f"Cut out {len(report.images)} image(s) for {report.document_id}; "
            f"{len(report.errors)} error(s)."
        )
        print(f"Output: {output_folder}")
        return 0 if report.status == "pass" else 1

    if args.command == "assemble-document":
        try:
            batch_root = args.batch.resolve()
            result = assemble_document(
                batch_root,
                args.document_id,
                args.split_report,
                args.check_report,
                args.ocr_output,
                args.image_output,
                args.output,
                args.description_report,
                args.skip_step7,
            )
        except (Step6Error, OSError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            _record_failed_run(args, run_id, "step-6")
            return 2

        record_step_run(
            batch_root,
            run_id,
            "step-6",
            "pass",
            [
                args.split_report
                or batch_root / "work" / "step-2-documents" / "split-report.json",
                args.check_report
                or batch_root / "work" / "step-1-checks" / "batch-report.json",
                args.ocr_output
                or batch_root / "work" / "step-4-ocr" / args.document_id,
                args.image_output
                or batch_root / "work" / "step-5-images" / args.document_id,
            ]
            + (
                []
                if args.skip_step7
                else [
                    args.description_report
                    or batch_root
                    / "work"
                    / "step-7-image-descriptions"
                    / f"{args.document_id}.json"
                ]
            ),
            [result.output_folder],
        )
        print(
            f"Assembled {result.document_id}: {len(result.pages)} page(s), "
            f"{len(result.images)} image(s)."
        )
        print(f"Bundle: {result.output_folder}")
        return 0

    if args.command == "describe-images":
        if not args.allow_cropped_image_upload:
            print(
                "ERROR: Vertex AI upload is disabled unless "
                "--allow-cropped-image-upload is provided.",
                file=sys.stderr,
            )
            return 2
        batch_root = args.batch.resolve()
        image_folder = (
            args.image_folder
            or batch_root / "work" / "step-5-images" / args.document_id
        ).resolve()
        cut_out_report = (
            args.cut_out_report
            or image_folder / "cut-out-report.json"
        ).resolve()
        output_path = (
            args.output
            or batch_root
            / "work"
            / "step-7-image-descriptions"
            / f"{args.document_id}.json"
        ).resolve()
        try:
            adapter = GeminiVertexAIAdapter(
                project=args.project,
                location=args.location,
                model=args.model,
            )
            report = describe_document_images(
                args.document_id,
                image_folder,
                cut_out_report,
                adapter,
            )
            write_image_description_report(report, output_path)
        except (Step7Error, OSError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            _record_failed_run(args, run_id, "step-7")
            return 2
        record_step_run(
            batch_root,
            run_id,
            "step-7",
            "pass" if report.status == "pass" else "fail",
            [cut_out_report, image_folder],
            [output_path],
            {
                "model": args.model,
                "project": args.project,
                "location": args.location,
                "external_crop_upload_consent": True,
            },
        )
        print(
            f"Described {len(report.images)} image(s); status: {report.status}."
        )
        print(f"Report: {output_path}")
        return 0 if report.status == "pass" else 1

    if args.command == "check-bundle":
        batch_root = args.batch.resolve()
        bundle_folder = (
            args.bundle
            or batch_root / "work" / "step-6-assemble" / args.document_id
        ).resolve()
        ocr_report = (
            args.ocr_report
            or batch_root / "work" / "step-4-ocr" / args.document_id / "ocr-report.json"
        ).resolve()
        output_path = (
            args.output
            or batch_root / "work" / "step-8-quality" / f"{args.document_id}.json"
        ).resolve()
        flag_log_path = (
            args.flag_log
            or batch_root / "work" / "logs" / "flag_log.csv"
        ).resolve()
        try:
            report = check_document_quality(
                bundle_folder=bundle_folder,
                document_id=args.document_id,
                ocr_report_path=ocr_report,
                vocabulary_path=args.vocabulary_file,
                vocabulary_threshold=args.vocabulary_threshold,
                run_id=run_id,
                flag_log_path=flag_log_path,
            )
            write_quality_report(report, output_path)
        except (Step8Error, OSError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            _record_failed_run(args, run_id, "step-8")
            return 2
        record_step_run(
            batch_root,
            run_id,
            "step-8",
            "pass" if report.bundle_status == "pass" else "fail",
            [bundle_folder, ocr_report]
            + ([args.vocabulary_file] if args.vocabulary_file else []),
            [output_path, flag_log_path],
            {"vocabulary_threshold": args.vocabulary_threshold},
        )
        print(
            f"Bundle {report.bundle_status}; {len(report.errors)} structural "
            f"error(s), {len(report.page_flags)} page/bundle flag(s)."
        )
        print(f"Report: {output_path}")
        return 0 if report.bundle_status == "pass" else 1

    if args.command == "publish-bundle":
        batch_root = args.batch.resolve()
        bundle_folder = (
            args.bundle
            or batch_root / "work" / "step-6-assemble" / args.document_id
        ).resolve()
        quality_report = (
            args.quality_report
            or batch_root / "work" / "step-8-quality" / f"{args.document_id}.json"
        ).resolve()
        storage_root = (args.storage_root or batch_root / "storage").resolve()
        try:
            result = publish_document_bundle(
                bundle_folder,
                args.document_id,
                quality_report,
                storage_root,
                run_id,
            )
        except (Step9Error, OSError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            _record_failed_run(args, run_id, "step-9")
            return 2
        record_step_run(
            batch_root,
            run_id,
            "step-9",
            "pass" if result.status == "published" else "fail",
            [bundle_folder, quality_report],
            [result.destination, storage_root / "logs"],
        )
        print(f"Bundle {result.status}: {result.destination}")
        return 0 if result.status == "published" else 1

    if args.command == "self-test":
        output_root = (
            args.output
            or Path.cwd() / f"ocr-pipeline-self-test-{run_id}"
        ).resolve()
        try:
            result = run_self_test(output_root, run_id)
        except (
            SelfTestError,
            RunTrackingError,
            BatchCheckError,
            Step2Error,
            Step3Error,
            Step4Error,
            Step5Error,
            Step6Error,
            Step7Error,
            Step8Error,
            Step9Error,
            ManifestError,
            OSError,
        ) as exc:
            print(f"ERROR: Synthetic pipeline test failed: {exc}", file=sys.stderr)
            return 1
        print(
            f"Synthetic end-to-end test passed: {len(result.checks)} checks."
        )
        print(f"Generated data and reports: {result.batch_root.parent}")
        print(f"Published bundle: {result.published_bundle}")
        return 0

    parser.error("unknown command")
    return 2
