#!/usr/bin/env python3
"""Small local web UI for the volume estimation pipeline.

This server intentionally uses only the Python standard library so it can run
inside the existing project environment without adding web dependencies.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import shutil
import shlex
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


AGENT_ROOT = Path(__file__).resolve().parents[1]
UI_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = UI_ROOT / "static"
RUN_ROOT = UI_ROOT / "runs"
DEFAULT_IMAGE_DIR = AGENT_ROOT / "realdata_anonymized" / "_anonymized" / "anon_0003"
DEFAULT_TEXT_FILE = DEFAULT_IMAGE_DIR / "003_items.txt"
DEFAULT_OUTPUT_DIR = AGENT_ROOT / "out_anycalib_unidepth" / "anon_0003_item"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEPTH_BACKENDS = ["anycalib_unidepth"]
SEG_BACKENDS = ["sam3"]
PDF_CUSTOMER_FIELDS = [
    ("customer_name", "고객명"),
    ("customer_phone", "전화번호"),
    ("origin_address", "출발지 주소"),
    ("inbound_date", "입고일"),
    ("inbound_start", "입고 시간대"),
    ("destination_address", "도착지 주소"),
    ("outbound_date", "출고일"),
    ("outbound_start", "출고 시간대"),
    ("inbound_work_type", "입고 작업"),
    ("outbound_work_type", "출고 작업"),
    ("request_text", "특이사항"),
]


@dataclass
class Job:
    id: str
    status: str
    command: str
    cwd: str
    output_dir: str
    image_dir: str
    log_path: Path
    started_at: float
    ended_at: float | None = None
    returncode: int | None = None
    error: str | None = None
    proc: subprocess.Popen | None = None

    def public(self, include_log: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "status": self.status,
            "command": self.command,
            "cwd": self.cwd,
            "output_dir": self.output_dir,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "returncode": self.returncode,
            "error": self.error,
            "elapsed_seconds": round((self.ended_at or time.time()) - self.started_at, 1),
        }
        if include_log:
            log_text = read_tail(self.log_path)
            data["log"] = log_text
            data["current_image"] = current_image_from_log(log_text, Path(self.image_dir))
        return data


JOB_LOCK = threading.Lock()
CURRENT_JOB: Job | None = None


def expand_path(value: str | None, default: Path) -> Path:
    if not value:
        return default
    expanded = os.path.expanduser(value.strip())
    path = Path(expanded)
    if not path.is_absolute():
        path = AGENT_ROOT / path
    return path.resolve(strict=False)


def read_tail(path: Path, limit: int = 90_000) -> str:
    if not path.exists():
        return ""
    size = path.stat().st_size
    with path.open("rb") as f:
        if size > limit:
            f.seek(size - limit)
        data = f.read()
    text = data.decode("utf-8", errors="replace")
    if size > limit:
        text = "... log truncated ...\n" + text
    return text


def json_response(handler: BaseHTTPRequestHandler, payload: Any, status: int = 200) -> None:
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


def text_response(handler: BaseHTTPRequestHandler, text: str, status: int = 200) -> None:
    raw = text.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/plain; charset=utf-8")
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


def list_images(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path), "files": [], "count": 0}
    if not path.is_dir():
        return {"exists": True, "is_dir": False, "path": str(path), "files": [], "count": 0}

    files = []
    for item in sorted(path.iterdir(), key=lambda p: p.name):
        if item.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        files.append(
            {
                "name": item.name,
                "stem": item.stem,
                "path": str(item),
                "size_bytes": item.stat().st_size,
            }
        )
    return {"exists": True, "is_dir": True, "path": str(path), "files": files, "count": len(files)}


def current_image_from_log(log_text: str, image_dir: Path) -> dict[str, Any] | None:
    matches = re.findall(r"\[Image\s+(\d+)/(\d+)\]\s+(.+?)\s+(?:→|->)", log_text)
    if not matches:
        return None

    idx, total, name = matches[-1]
    image_path = image_dir / name.strip()
    payload: dict[str, Any] = {
        "index": int(idx),
        "total": int(total),
        "name": name.strip(),
        "path": str(image_path) if image_path.exists() else None,
    }
    return payload


def list_output_images(output_dir: Path, limit: int = 300) -> list[dict[str, Any]]:
    if not output_dir.exists() or not output_dir.is_dir():
        return []

    files: list[dict[str, Any]] = []
    for item in sorted(output_dir.rglob("*"), key=lambda p: p.relative_to(output_dir).as_posix()):
        if len(files) >= limit:
            break
        if not item.is_file() or item.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if item.name != "depth_with_objects.png":
            continue
        files.append(
            {
                "name": item.name,
                "relative_path": item.relative_to(output_dir).as_posix(),
                "path": str(item),
                "size_bytes": item.stat().st_size,
            }
        )
    return files


def load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_truck_name(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("5톤 초과"):
        return "5톤"
    return value


def summarize_results(output_dir: Path) -> dict[str, Any]:
    batch_path = output_dir / "batch_summary.json"
    quote_pdf = output_dir / "quote.pdf"
    quote_pdf_path = str(quote_pdf) if quote_pdf.exists() else None
    output_images = list_output_images(output_dir)
    batch = load_json(batch_path)
    if batch is not None:
        files = []
        for item in batch.get("files", []):
            files.append(
                {
                    "file": Path(item.get("file", "")).name,
                    "path": item.get("file"),
                    "output_dir": item.get("output_dir"),
                    "volume_m3": item.get("total_volume_m3", 0),
                    "raw_volume_m3": item.get("raw_total_volume_m3", item.get("total_volume_m3", 0)),
                    "num_detected_objects": item.get("num_detected_objects", 0),
                    "detected_categories": item.get("detected_categories", []),
                    "category_counts": item.get("category_counts", {}),
                    "elapsed_seconds": item.get("elapsed_seconds", 0),
                    "processing_seconds": item.get("processing_seconds", 0),
                    "save_seconds": item.get("save_seconds", 0),
                }
            )
        return {
            "exists": True,
            "kind": "batch",
            "path": str(batch_path),
            "quote_pdf": quote_pdf_path,
            "output_images": output_images,
            "summary": {
                "input_dir": batch.get("input_dir"),
                "num_files": batch.get("num_files", 0),
                "num_processed": batch.get("num_processed", 0),
                "num_failed": batch.get("num_failed", 0),
                "total_volume_m3": batch.get("total_volume_m3", 0),
                "raw_total_volume_m3": batch.get("raw_total_volume_m3", 0),
                "total_cbm_floor": batch.get("total_cbm_floor", 0),
                "recommended_truck": normalize_truck_name(batch.get("recommended_truck")),
                "total_elapsed_seconds": batch.get("total_elapsed_seconds", 0),
                "total_processing_seconds": batch.get("total_processing_seconds", 0),
                "total_save_seconds": batch.get("total_save_seconds", 0),
                "avg_processing_seconds_per_file": batch.get("avg_processing_seconds_per_file", 0),
                "avg_save_seconds_per_file": batch.get("avg_save_seconds_per_file", 0),
                "category_counts": batch.get("category_counts", {}),
                "detected_categories": batch.get("detected_categories", []),
                "files": files,
            },
        }

    result_path = output_dir / "result.json"
    result = load_json(result_path)
    if result is not None:
        objects = [
            {
                "object": obj.get("object"),
                "object_category": obj.get("object_category"),
                "volume_m3": obj.get("volume_m3", 0),
                "dimensions_m": obj.get("dimensions_m"),
                "detection_score": obj.get("detection_score"),
                "error": obj.get("error"),
            }
            for obj in result.get("objects", [])
        ]
        return {
            "exists": True,
            "kind": "single",
            "path": str(result_path),
            "quote_pdf": quote_pdf_path,
            "output_images": output_images,
            "summary": {
                "total_volume_m3": result.get("total_volume_m3", 0),
                "raw_total_volume_m3": result.get("raw_total_volume_m3", result.get("total_volume_m3", 0)),
                "elapsed_seconds": result.get("elapsed_seconds", 0),
                "processing_seconds": result.get("processing_seconds", 0),
                "save_seconds": result.get("save_seconds", 0),
                "objects": objects,
            },
        }

    return {
        "exists": bool(output_images or quote_pdf_path),
        "path": str(output_dir),
        "quote_pdf": quote_pdf_path,
        "output_images": output_images,
    }


def format_display_command(args: list[str]) -> str:
    return "\n".join(
        [
            "cd " + shlex.quote(str(AGENT_ROOT)),
            "conda activate volume_est",
            shlex.join(args),
        ]
    )


def pdf_customer_payload(payload: dict[str, Any]) -> dict[str, str]:
    raw = payload.get("pdf_customer") or {}
    if not isinstance(raw, dict):
        return {}

    customer: dict[str, str] = {}
    has_value = False
    for key, _ in PDF_CUSTOMER_FIELDS:
        value = raw.get(key, "")
        text = str(value).strip() if value is not None else ""
        if text:
            has_value = True
        customer[key] = text

    if has_value and not customer.get("request_text"):
        customer["request_text"] = "없음"
    return customer if has_value else {}


def write_pdf_customer_json(payload: dict[str, Any], job_id: str) -> Path | None:
    customer = pdf_customer_payload(payload)
    if not customer:
        return None
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    customer_path = RUN_ROOT / f"{job_id}_customer.json"
    customer_path.write_text(
        json.dumps(customer, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return customer_path


def build_pipeline_args(payload: dict[str, Any]) -> tuple[list[str], Path, Path, str]:
    image_dir = expand_path(payload.get("image_dir"), DEFAULT_IMAGE_DIR)
    text_file = expand_path(payload.get("text_file"), DEFAULT_TEXT_FILE)
    output_dir = expand_path(payload.get("output_dir"), DEFAULT_OUTPUT_DIR)

    depth_backend = payload.get("depth_backend") or "anycalib_unidepth"
    seg_backend = payload.get("seg_backend") or "sam3"
    if depth_backend not in DEPTH_BACKENDS:
        raise ValueError(f"Unsupported depth backend: {depth_backend}")
    if seg_backend not in SEG_BACKENDS:
        raise ValueError(f"Unsupported segmentation backend: {seg_backend}")

    args = [
        "python",
        "-u",
        "pipeline.py",
        "--image-dir",
        str(image_dir),
        "--text-file",
        str(text_file),
        "--depth-backend",
        depth_backend,
        "--seg-backend",
        seg_backend,
        "--output",
        str(output_dir),
        "--json-only",
        "--pdf",
    ]

    display_command = format_display_command(args)
    return args, output_dir, image_dir, display_command


def clear_output_dir(output_dir: Path) -> None:
    try:
        relative = output_dir.relative_to(AGENT_ROOT)
    except ValueError as exc:
        raise ValueError("Output folder must be inside the agent project.") from exc

    if not relative.parts or not relative.parts[0].startswith("out"):
        raise ValueError("Output folder cleanup is only allowed for agent/out... paths.")

    if output_dir.exists() and not output_dir.is_dir():
        output_dir.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in output_dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def start_job(payload: dict[str, Any]) -> Job:
    global CURRENT_JOB

    args, output_dir, image_dir, display_command = build_pipeline_args(payload)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    job_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    log_path = RUN_ROOT / f"{job_id}.log"
    customer_json_path = write_pdf_customer_json(payload, job_id)
    if customer_json_path is not None:
        args.extend(["--pdf-customer-json", str(customer_json_path)])
        display_command = format_display_command(args)

    quoted_pipeline = shlex.join(args)
    shell_script = (
        "set -e; "
        "if [ -f /opt/conda/etc/profile.d/conda.sh ]; then "
        "source /opt/conda/etc/profile.d/conda.sh; "
        "fi; "
        "conda activate volume_est; "
        f"exec {quoted_pipeline}"
    )

    job = Job(
        id=job_id,
        status="running",
        command=display_command,
        cwd=str(AGENT_ROOT),
        output_dir=str(output_dir),
        image_dir=str(image_dir),
        log_path=log_path,
        started_at=time.time(),
    )

    with JOB_LOCK:
        if CURRENT_JOB and CURRENT_JOB.status == "running":
            raise RuntimeError("Another pipeline job is already running.")
        clear_output_dir(output_dir)
        CURRENT_JOB = job

    def worker() -> None:
        with log_path.open("w", encoding="utf-8", errors="replace") as log:
            log.write(display_command + "\n\n")
            log.flush()
            try:
                proc = subprocess.Popen(
                    ["/bin/bash", "-lc", shell_script],
                    cwd=str(AGENT_ROOT),
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                job.proc = proc
                rc = proc.wait()
                job.returncode = rc
                if job.status == "stopping":
                    job.status = "stopped"
                else:
                    job.status = "completed" if rc == 0 else "failed"
            except Exception as exc:  # noqa: BLE001 - show local operator the real issue.
                job.error = str(exc)
                job.status = "failed"
                log.write(f"\nERROR: {exc}\n")
            finally:
                job.ended_at = time.time()
                log.write(f"\n[agent_ui] status={job.status} returncode={job.returncode}\n")

    threading.Thread(target=worker, daemon=True).start()
    return job


def stop_job() -> dict[str, Any]:
    with JOB_LOCK:
        job = CURRENT_JOB
    if not job or job.status != "running" or job.proc is None:
        return {"stopped": False, "message": "No running job."}
    job.status = "stopping"
    try:
        os.killpg(job.proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return {"stopped": False, "message": "Process already exited."}
    return {"stopped": True, "message": "Stop signal sent."}


class AgentUIHandler(BaseHTTPRequestHandler):
    server_version = "AgentUI/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[agent_ui] {self.client_address[0]} - " + fmt % args)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API.
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == "/api/defaults":
            return json_response(
                self,
                {
                    "agent_root": str(AGENT_ROOT),
                    "image_dir": str(DEFAULT_IMAGE_DIR),
                    "output_dir": str(DEFAULT_OUTPUT_DIR),
                    "text_file": str(DEFAULT_TEXT_FILE),
                    "depth_backends": DEPTH_BACKENDS,
                    "seg_backends": SEG_BACKENDS,
                    "pdf_customer_fields": [
                        {"key": key, "label": label}
                        for key, label in PDF_CUSTOMER_FIELDS
                    ],
                },
            )
        if path == "/api/images":
            image_dir = expand_path(query.get("path", [None])[0], DEFAULT_IMAGE_DIR)
            return json_response(self, list_images(image_dir))
        if path == "/api/results":
            output_dir = expand_path(query.get("output", [None])[0], DEFAULT_OUTPUT_DIR)
            try:
                return json_response(self, summarize_results(output_dir))
            except json.JSONDecodeError as exc:
                return json_response(self, {"exists": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
        if path == "/api/download":
            target = query.get("path", [None])[0]
            if not target:
                return text_response(self, "Missing path", HTTPStatus.BAD_REQUEST)
            return self.serve_download(target)
        if path == "/api/job":
            with JOB_LOCK:
                job = CURRENT_JOB
            payload = {"job": job.public() if job else None}
            if job:
                payload["results"] = summarize_results(Path(job.output_dir))
            return json_response(self, payload)

        self.serve_static(path)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API.
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            return json_response(self, {"error": str(exc)}, HTTPStatus.BAD_REQUEST)

        if parsed.path == "/api/run":
            try:
                job = start_job(payload)
            except Exception as exc:  # noqa: BLE001 - local web UI should surface exact error.
                return json_response(self, {"error": str(exc)}, HTTPStatus.CONFLICT)
            return json_response(self, {"job": job.public(include_log=False)})

        if parsed.path == "/api/stop":
            return json_response(self, stop_job())

        return json_response(self, {"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def serve_download(self, raw_path: str) -> None:
        file_path = expand_path(raw_path, AGENT_ROOT)
        try:
            file_path.relative_to(AGENT_ROOT)
        except ValueError:
            return text_response(self, "Forbidden", HTTPStatus.FORBIDDEN)
        if not file_path.exists() or not file_path.is_file():
            return text_response(self, "Not found", HTTPStatus.NOT_FOUND)

        content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        raw = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        disposition = "inline" if file_path.suffix.lower() in IMAGE_SUFFIXES | {".pdf"} else "attachment"
        self.send_header("Content-Disposition", f'{disposition}; filename="{file_path.name}"')
        self.end_headers()
        self.wfile.write(raw)

    def serve_static(self, request_path: str) -> None:
        if request_path in {"", "/"}:
            file_path = STATIC_ROOT / "index.html"
        else:
            clean = request_path.lstrip("/")
            if clean.startswith("static/"):
                clean = clean[len("static/") :]
            file_path = (STATIC_ROOT / clean).resolve(strict=False)

        try:
            file_path.relative_to(STATIC_ROOT)
        except ValueError:
            return text_response(self, "Forbidden", HTTPStatus.FORBIDDEN)

        if not file_path.exists() or not file_path.is_file():
            return text_response(self, "Not found", HTTPStatus.NOT_FOUND)

        content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        raw = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local Agent volume web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=7860, type=int)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), AgentUIHandler)
    print(f"Agent UI: http://{args.host}:{args.port}")
    print(f"Project: {AGENT_ROOT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
