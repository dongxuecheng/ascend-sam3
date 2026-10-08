#!/usr/bin/env python3
"""Benchmark the SAM3 HTTP service with all images in a directory.

The script uses only the Python standard library. It builds multipart/form-data
requests itself, so filenames containing spaces, commas, semicolons, or non-ASCII
characters do not depend on curl's form parser behavior.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Sequence


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class PreparedRequest:
    image_path: str
    body: bytes
    content_type: str


@dataclass
class RequestResult:
    image_path: str
    status: int
    elapsed_seconds: float
    upstream: str
    worker_pid: str = ""
    error: str = ""
    refinement: dict | None = None
    detections: list | None = None

    @property
    def success(self) -> bool:
        return 200 <= self.status < 300 and not self.error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark SAM3 ordinary or obj-refine file inference with a directory of images."
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:18000/predict/file",
        help="SAM3 file endpoint; default path switches automatically in obj-refine mode.",
    )
    parser.add_argument(
        "--images",
        default="test-images",
        help="Directory containing test images (default: test-images).",
    )
    parser.add_argument(
        "--concurrency", type=int, default=2, help="Concurrent HTTP requests."
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Number of times to send every image.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Warm-up requests excluded from statistics.",
    )
    parser.add_argument(
        "--class-name",
        action="append",
        dest="class_names",
        help="Class prompt; repeat this option for multiple classes.",
    )
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument("--mode", choices=["detect", "obj-refine"], default="detect")
    parser.add_argument("--pre-detect-label", action="append", dest="pre_detect_labels")
    parser.add_argument("--pre-detect-confidence", type=float)
    parser.add_argument("--max-crops", type=int, help="Crop budget; omitted uses the server's .env default.")
    parser.add_argument("--crop-max-size", type=int, default=640)
    parser.add_argument("--no-merge-results", action="store_true")
    parser.add_argument("--save-results", action="store_true", help="Include boxes/RLE in JSON for manual accuracy review.")
    parser.add_argument(
        "--return-mask",
        action="store_true",
        help="Request mask generation and transfer (disabled by default).",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--label", default="sam3-benchmark")
    parser.add_argument(
        "--json-output",
        help="Optional path for a machine-readable JSON report.",
    )
    args = parser.parse_args()

    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if args.rounds < 1:
        parser.error("--rounds must be at least 1")
    if args.warmup < 0:
        parser.error("--warmup must not be negative")
    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be in [0, 1]")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if (args.max_crops is not None and not 1 <= args.max_crops <= 32) or not 1 <= args.crop_max_size <= 16384:
        parser.error("--max-crops must be in [1,32], --crop-max-size in [1,16384]")
    if args.pre_detect_confidence is not None and not 0 <= args.pre_detect_confidence <= 1:
        parser.error("--pre-detect-confidence must be in [0,1]")

    if not args.class_names:
        args.class_names = ["helmet"] if args.mode == "obj-refine" else ["person", "fire"]
    args.pre_detect_labels = args.pre_detect_labels or ["person"]
    if args.mode == "obj-refine" and args.url.endswith("/predict/file"):
        args.url = args.url[:-len("/predict/file")] + "/predict-obj-refine/file"
    return args


def discover_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise ValueError(f"Image directory does not exist: {directory}")

    images = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError(f"No supported image files found under: {directory}")
    return images


def _multipart_line(buffer: bytearray, value: str) -> None:
    buffer.extend(value.encode("utf-8"))
    buffer.extend(b"\r\n")


def build_multipart_request(
    image_path: Path,
    class_names: Sequence[str],
    confidence: float,
    return_mask: bool,
    extra_fields: Sequence[tuple[str, str]] = (),
) -> PreparedRequest:
    boundary = f"----sam3-benchmark-{uuid.uuid4().hex}"
    body = bytearray()

    fields: list[tuple[str, str]] = [
        *(('class_names', name) for name in class_names),
        ("confidence", str(confidence)),
        ("return_mask", "true" if return_mask else "false"),
        *extra_fields,
    ]
    for name, value in fields:
        _multipart_line(body, f"--{boundary}")
        _multipart_line(body, f'Content-Disposition: form-data; name="{name}"')
        _multipart_line(body, "")
        _multipart_line(body, value)

    mime_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    safe_filename = f"benchmark{image_path.suffix.lower()}"
    _multipart_line(body, f"--{boundary}")
    _multipart_line(
        body,
        f'Content-Disposition: form-data; name="image"; filename="{safe_filename}"',
    )
    _multipart_line(body, f"Content-Type: {mime_type}")
    _multipart_line(body, "")
    body.extend(image_path.read_bytes())
    body.extend(b"\r\n")
    _multipart_line(body, f"--{boundary}--")

    return PreparedRequest(
        image_path=str(image_path),
        body=bytes(body),
        content_type=f"multipart/form-data; boundary={boundary}",
    )


def execute_request(
    url: str, prepared: PreparedRequest, timeout: float, save_results: bool = False
) -> RequestResult:
    request = urllib.request.Request(
        url,
        data=prepared.body,
        headers={
            "Content-Type": prepared.content_type,
            "Content-Length": str(len(prepared.body)),
            "User-Agent": "ascend-sam3-benchmark/1.0",
        },
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())
            if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
                raise ValueError("Response must contain a results list")
            elapsed = time.perf_counter() - started
            return RequestResult(
                image_path=prepared.image_path,
                status=response.status,
                elapsed_seconds=elapsed,
                upstream=response.headers.get("X-SAM3-Upstream", "direct"),
                worker_pid=response.headers.get("X-SAM3-Worker-PID", ""),
                refinement=payload.get("refinement"),
                detections=payload["results"] if save_results else None,
            )
    except urllib.error.HTTPError as exc:
        detail = exc.read(512).decode("utf-8", errors="replace")
        return RequestResult(
            image_path=prepared.image_path,
            status=exc.code,
            elapsed_seconds=time.perf_counter() - started,
            upstream=exc.headers.get("X-SAM3-Upstream", "") if exc.headers else "",
            worker_pid=(
                exc.headers.get("X-SAM3-Worker-PID", "") if exc.headers else ""
            ),
            error=detail or str(exc),
        )
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        return RequestResult(
            image_path=prepared.image_path,
            status=0,
            elapsed_seconds=time.perf_counter() - started,
            upstream="",
            worker_pid="",
            error=str(exc),
        )


def percentile(sorted_values: Sequence[float], quantile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (len(sorted_values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] + (
        sorted_values[upper] - sorted_values[lower]
    ) * fraction


def make_summary(
    args: argparse.Namespace,
    images: Sequence[Path],
    results: Sequence[RequestResult],
    wall_seconds: float,
) -> dict:
    successful = [result for result in results if result.success]
    latencies = sorted(result.elapsed_seconds for result in successful)
    upstream_counts = Counter(
        result.upstream or "unknown" for result in successful
    )
    worker_counts = Counter(
        f"{result.upstream or 'unknown'} pid={result.worker_pid or 'unknown'}"
        for result in successful
    )

    summary = {
        "label": args.label,
        "url": args.url,
        "image_directory": str(Path(args.images).resolve()),
        "unique_images": len(images),
        "rounds": args.rounds,
        "concurrency": args.concurrency,
        "class_names": args.class_names,
        "mode": args.mode,
        "pre_detect_labels": args.pre_detect_labels if args.mode == "obj-refine" else [],
        "merge_results": not args.no_merge_results if args.mode == "obj-refine" else None,
        "return_mask": args.return_mask,
        "total_requests": len(results),
        "successful_requests": len(successful),
        "failed_requests": len(results) - len(successful),
        "wall_seconds": wall_seconds,
        "throughput_images_per_second": (
            len(successful) / wall_seconds if wall_seconds > 0 else 0.0
        ),
        "latency_seconds": {
            "min": min(latencies) if latencies else 0.0,
            "mean": mean(latencies) if latencies else 0.0,
            "p50": median(latencies) if latencies else 0.0,
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
            "max": max(latencies) if latencies else 0.0,
        },
        "upstream_counts": dict(sorted(upstream_counts.items())),
        "worker_counts": dict(sorted(worker_counts.items())),
    }
    refine_stats = [r.refinement for r in successful if r.refinement is not None]
    if refine_stats:
        summary["refinement"] = {
            "mean_crops_processed": mean(r["crops_processed"] for r in refine_stats),
            "limited_requests": sum(bool(r["limited"]) for r in refine_stats),
            "mean_stage_ms": {
                key: mean(r["timings_ms"][key] for r in refine_stats)
                for key in refine_stats[0]["timings_ms"]
            },
        }
    return summary


def print_summary(summary: dict) -> None:
    latency = summary["latency_seconds"]
    print("\n=== SAM3 benchmark summary ===")
    print(f"label:       {summary['label']}")
    print(f"requests:    {summary['successful_requests']}/{summary['total_requests']} succeeded")
    print(f"wall:        {summary['wall_seconds']:.3f} s")
    print(
        "throughput:  "
        f"{summary['throughput_images_per_second']:.3f} images/s"
    )
    print(
        "latency:     "
        f"mean={latency['mean']:.3f}s "
        f"p50={latency['p50']:.3f}s "
        f"p95={latency['p95']:.3f}s "
        f"p99={latency['p99']:.3f}s "
        f"min={latency['min']:.3f}s "
        f"max={latency['max']:.3f}s"
    )
    print("upstreams:")
    for upstream, count in summary["upstream_counts"].items():
        print(f"  {upstream}: {count}")
    print("workers:")
    for worker, count in summary["worker_counts"].items():
        print(f"  {worker}: {count}")
    if "refinement" in summary:
        print("refinement: " + json.dumps(summary["refinement"], ensure_ascii=False))


def write_json_report(
    output_path: Path,
    args: argparse.Namespace,
    summary: dict,
    results: Iterable[RequestResult],
) -> None:
    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "arguments": vars(args),
        "summary": summary,
        "results": [asdict(result) for result in results],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    try:
        images = discover_images(Path(args.images))
        extra_fields = []
        if args.mode == "obj-refine":
            crop_cfg = {"max_size": args.crop_max_size}
            if args.max_crops is not None:
                crop_cfg["max_crops"] = args.max_crops
            extra_fields = [
                *(("pre_detect_labels", label) for label in args.pre_detect_labels),
                ("merge_results", "false" if args.no_merge_results else "true"),
                ("crop_config_json", json.dumps(crop_cfg)),
            ]
            if args.pre_detect_confidence is not None:
                extra_fields.append(("pre_detect_confidence", str(args.pre_detect_confidence)))
        prepared_images = [
            build_multipart_request(
                image,
                args.class_names,
                args.confidence,
                args.return_mask,
                extra_fields,
            )
            for image in images
        ]
    except (OSError, ValueError) as exc:
        print(f"benchmark setup failed: {exc}", file=sys.stderr)
        return 2

    print(
        f"Prepared {len(images)} image(s), rounds={args.rounds}, "
        f"concurrency={args.concurrency}, warmup={args.warmup}"
    )

    for warmup_index in range(args.warmup):
        prepared = prepared_images[warmup_index % len(prepared_images)]
        result = execute_request(args.url, prepared, args.timeout, args.save_results)
        if not result.success:
            print(
                f"warmup failed for {result.image_path}: "
                f"status={result.status} error={result.error}",
                file=sys.stderr,
            )
            return 2

    requests = prepared_images * args.rounds
    results: list[RequestResult] = []
    progress_step = max(1, len(requests) // 10)
    wall_started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as executor:
        futures = [
            executor.submit(execute_request, args.url, prepared, args.timeout, args.save_results)
            for prepared in requests
        ]
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            result = future.result()
            results.append(result)
            if completed % progress_step == 0 or completed == len(requests):
                print(f"progress: {completed}/{len(requests)}")
    wall_seconds = time.perf_counter() - wall_started

    summary = make_summary(args, images, results, wall_seconds)
    print_summary(summary)

    failures = [result for result in results if not result.success]
    if failures:
        print("failures:", file=sys.stderr)
        for result in failures:
            print(
                f"  {result.image_path}: status={result.status} "
                f"elapsed={result.elapsed_seconds:.3f}s error={result.error}",
                file=sys.stderr,
            )

    if args.json_output:
        write_json_report(Path(args.json_output), args, summary, results)
        print(f"JSON report: {args.json_output}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
