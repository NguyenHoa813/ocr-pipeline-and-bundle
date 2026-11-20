import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal


RunStatus = Literal["pass", "fail", "skipped", "stale"]
CONFIG_VERSION = "1"
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}")
INVALIDATED_BY_STEP = {
    "step-1": (
        "step-2", "step-3", "step-4", "step-4.1", "step-5",
        "step-6", "step-7", "step-8", "step-9",
    ),
    "step-2": (
        "step-3", "step-4", "step-4.1", "step-5", "step-6",
        "step-7", "step-8", "step-9",
    ),
    "step-3": (
        "step-4", "step-4.1", "step-5", "step-6", "step-7",
        "step-8", "step-9",
    ),
    "step-4": ("step-4.1", "step-5", "step-6", "step-7", "step-8", "step-9"),
    "step-4.1": ("step-6", "step-8", "step-9"),
    "step-5": ("step-6", "step-7", "step-8", "step-9"),
    "step-6": ("step-8", "step-9"),
    "step-7": ("step-6", "step-8", "step-9"),
    "step-8": ("step-9",),
    "step-9": (),
}


class RunTrackingError(Exception):
    """Raised when run provenance cannot be recorded."""


def new_run_id() -> str:
    return f"run-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    if not path.is_dir():
        raise RunTrackingError(f"Run input/output does not exist: {path}")
    for child in sorted(
        (item for item in path.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(path).as_posix().casefold(),
    ):
        relative = child.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with child.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def record_step_run(
    batch_root: Path,
    run_id: str,
    step: str,
    status: RunStatus,
    input_paths: list[Path],
    output_paths: list[Path],
    configuration: dict[str, object] | None = None,
) -> Path:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise RunTrackingError(f"Invalid run ID: {run_id}")
    if not re.fullmatch(r"step-(?:[1-9]|4\.1)", step):
        raise RunTrackingError(f"Invalid pipeline step name: {step}")
    batch_root = Path(batch_root).resolve()
    run_folder = batch_root / "work" / "runs"
    run_folder.mkdir(parents=True, exist_ok=True)
    run_path = run_folder / f"{run_id}.json"
    if run_path.exists():
        try:
            run_data = json.loads(run_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RunTrackingError(f"Cannot read run record {run_path}: {exc}") from exc
        if not isinstance(run_data, dict) or run_data.get("run_id") != run_id:
            raise RunTrackingError(f"Invalid run record: {run_path}")
    else:
        run_data = {
            "schema_version": 1,
            "run_id": run_id,
            "batch_root": str(batch_root),
            "config_version": CONFIG_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "steps": {},
        }
    inputs: dict[str, str] = {}
    outputs: dict[str, str] = {}
    config_paths = [
        batch_root / "config.json",
        batch_root / "batch_manifest.yaml",
    ]
    for path in [*input_paths, *config_paths]:
        resolved = Path(path).resolve()
        if resolved.exists():
            inputs[str(resolved)] = _checksum(resolved)
    for path in output_paths:
        resolved = Path(path).resolve()
        if resolved.exists():
            outputs[str(resolved)] = _checksum(resolved)
    steps = run_data.get("steps")
    if not isinstance(steps, dict):
        raise RunTrackingError(f"Run record steps must be an object: {run_path}")
    previous_step = steps.get(step)
    if isinstance(previous_step, dict) and any(
        previous_step.get(key) != value
        for key, value in (
            ("input_sha256", inputs),
            ("output_sha256", outputs),
            ("configuration", configuration or {}),
        )
    ):
        for downstream_step in INVALIDATED_BY_STEP[step]:
            downstream = steps.get(downstream_step)
            if isinstance(downstream, dict) and downstream.get("status") != "stale":
                downstream["status"] = "stale"
                downstream["stale_reason"] = (
                    f"{step} inputs, outputs, or configuration changed."
                )
    steps[step] = {
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "input_sha256": inputs,
        "output_sha256": outputs,
        "configuration": configuration or {},
    }
    temporary_path = run_path.with_suffix(".tmp")
    try:
        temporary_path.write_text(
            json.dumps(run_data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, run_path)
    except OSError as exc:
        if temporary_path.exists():
            temporary_path.unlink()
        raise RunTrackingError(f"Cannot write run record {run_path}: {exc}") from exc
    return run_path
