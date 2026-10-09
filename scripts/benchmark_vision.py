#!/usr/bin/env python3
"""Run isolated Vision baseline/candidate processes; no third-party Python packages."""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import shutil
import statistics
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from vision_workflow import (ROOT, container_path, digest, docker_args, ensure_idle, image_identity,
                             logged_run, new_run, settings, workspace_path, write_json)

SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
STAGES = ("sampling_ms", "upload_ms", "inference_ms", "total_ms")
FEATURE_BYTES_PER_CASE = 4 * 256 * (288*288 + 144*144 + 72*72)


def discover_cases(images: Path, crops: Path | None = None, max_images: int | None = None,
                   root: Path = ROOT) -> list[dict]:
    if not images.is_dir():
        raise ValueError(f"Image directory missing: {images}")
    paths = sorted(path.resolve() for path in images.iterdir()
                   if path.is_file() and path.suffix.lower() in SUFFIXES)
    if max_images is not None:
        paths = paths[:max_images]
    if not paths:
        raise ValueError("No supported images found")
    cases = [{"image": str(workspace_path(path, root)), "roi": None} for path in paths]
    if crops is not None:
        entries = json.loads(crops.read_text(encoding="utf-8-sig"))
        if not isinstance(entries, list):
            raise ValueError("Crop manifest must be a JSON array")
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"image", "roi"}:
                raise ValueError("Each crop requires exactly image and roi")
            if not isinstance(entry["image"], str):
                raise ValueError("Crop image must be a path")
            path = workspace_path(entry["image"], root)
            if not path.is_file():
                raise ValueError(f"Crop source missing: {path}")
            roi = entry["roi"]
            if (not isinstance(roi, list) or len(roi) != 4 or
                    any(type(v) is not int or v > 2147483647 for v in roi) or
                    min(roi[:2]) < 0 or min(roi[2:]) <= 0):
                raise ValueError("ROI requires integer [x,y,width,height], positive size")
            cases.append({"image": str(path), "roi": roi})
    return cases


def summary(values: list[float]) -> dict:
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Missing/invalid benchmark samples")
    ordered = sorted(values)
    def percentile(q):
        index = q*(len(ordered)-1)
        lower = math.floor(index)
        upper = min(lower+1, len(ordered)-1)
        return ordered[lower] + (ordered[upper]-ordered[lower])*(index-lower)
    return {"mean": statistics.mean(values), "p50": percentile(.5), "p95": percentile(.95),
            "min": ordered[0], "max": ordered[-1], "samples": len(values)}


def aggregate(reports: list[dict]) -> dict:
    stages = {stage: [] for stage in STAGES}
    by_mode = {"full": {stage: [] for stage in STAGES}, "crop": {stage: [] for stage in STAGES}}
    for report in reports:
        for case in report["cases"]:
            mode = "full" if case["roi"] is None else "crop"
            for sample in case["samples"]:
                if len(sample) != len(STAGES):
                    raise ValueError("Invalid native timing row")
                for stage, value in zip(STAGES, sample):
                    stages[stage].append(value)
                    by_mode[mode][stage].append(value)
    return {"all": {stage: summary(values) for stage, values in stages.items()},
            **{mode: {stage: summary(values) for stage, values in data.items()}
               for mode, data in by_mode.items() if data["total_ms"]}}


def speed_comparison(baseline: dict, candidate: dict) -> dict:
    result = {}
    for mode in baseline:
        result[mode] = {}
        for stage in STAGES:
            before = baseline[mode][stage]["mean"]
            after = candidate[mode][stage]["mean"]
            result[mode][stage] = {"baseline_ms": before, "candidate_ms": after,
                                  "latency_reduction_percent": (1-after/before)*100 if before else None,
                                  "speedup": before/after if after else None}
    return result


def parse_args() -> argparse.Namespace:
    config = settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default=config.get("VISION_MODEL", "models/om-models/vision-encoder-trt.om"))
    parser.add_argument("--candidate", help="Omit for baseline-only performance profiling")
    parser.add_argument("--images", default="test-images")
    parser.add_argument("--crops", help="JSON array of {image: project-relative path, roi: [x,y,w,h]}")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--device", type=int, default=config.get("ASCEND_PHYSICAL_DEVICE_ID", "2"))
    parser.add_argument("--image", default=config.get("SAM3_BENCH_IMAGE", "ascend-sam3-service:latest"))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=2, help="Alternate AB/BA order between repeats")
    parser.add_argument("--atol", type=float, default=.001)
    parser.add_argument("--rtol", type=float, default=.001)
    parser.add_argument("--output-dir", default="benchmark-results")
    parser.add_argument("--keep-features", action="store_true")
    parser.add_argument("--profile", action="store_true", help="msprof capture; timings are diagnostic, not speed claims")
    parser.add_argument("--allow-privileged-peer", action="store_true", help="Confirm privileged peers use only other devices; never bypasses explicit same-device mapping")
    parser.add_argument("--local", action="store_true", help="Use locally compiled executable without Docker")
    parser.add_argument("--binary", default="build/ascendsam3_vision_bench", help="Native path, only with --local")
    args = parser.parse_args()
    if args.device < 0 or args.warmup < 0 or args.iterations < 1 or args.repeats < 1:
        parser.error("Device/warmup must be >=0; iterations/repeats must be >=1")
    if args.max_images is not None and args.max_images < 1:
        parser.error("--max-images must be >=1")
    if any(not math.isfinite(v) or v < 0 for v in (args.atol, args.rtol)):
        parser.error("Finite nonnegative feature tolerances required")
    return args


def native_command(args: argparse.Namespace, model: Path, manifest: Path, report: Path,
                   features: Path | None = None, reference: Path | None = None) -> list[str]:
    translate = str if args.local else container_path
    # This executable belongs to the image; project data is mounted at /workspace.
    binary = str(workspace_path(args.binary)) if args.local else "/app/bin/ascendsam3_vision_bench"
    command = [binary, "--model", translate(model), "--manifest", translate(manifest),
               "--json-output", translate(report), "--device", str(args.device if args.local else 0),
               "--warmup", str(args.warmup), "--iterations", str(args.iterations),
               "--atol", str(args.atol), "--rtol", str(args.rtol)]
    if features is not None:
        command += ["--features-dir", translate(features)]
    if reference is not None:
        command += ["--reference-dir", translate(reference)]
    if args.profile:
        command = ["msprof", f"--output={translate(report.parent / (report.stem+'-profile'))}", *command]
    if not args.local:
        command = docker_args(args.device, args.image, "/usr/bin/env") + ["-u", "ASCEND_RT_VISIBLE_DEVICES", *command]
    return command


def validate_report(report: dict, cases: list[dict], args: argparse.Namespace) -> None:
    if report.get("schema_version") != 1 or len(report.get("cases", [])) != len(cases):
        raise ValueError("Incomplete/incompatible native report")
    for actual, expected in zip(report["cases"], cases):
        path = str(Path(expected["image"])) if args.local else container_path(Path(expected["image"]))
        if actual["image"] != path or actual["roi"] != expected["roi"] or len(actual["samples"]) != args.iterations:
            raise ValueError("Native report case/sample mismatch")
        features = actual.get("features", [])
        if len(features) != 3 or [v["index"] for v in features] != [0, 1, 2]:
            raise ValueError("Incomplete feature check")
    aggregate([report])  # Validate every raw timing, not only precomputed summaries.


def run(args: argparse.Namespace) -> int:
    baseline = workspace_path(args.baseline)
    candidate = workspace_path(args.candidate) if args.candidate else None
    for model in (baseline, candidate):
        if model is not None and (not model.is_file() or model.stat().st_size == 0):
            raise ValueError(f"Model missing/empty: {model}")
    if candidate == baseline:
        raise ValueError("Candidate and baseline must be different files")
    cases = discover_cases(workspace_path(args.images), workspace_path(args.crops) if args.crops else None, args.max_images)
    parent = workspace_path(args.output_dir)
    if not args.local:
        ensure_idle(args.device, allow_privileged=getattr(args, "allow_privileged_peer", False))
        image_info = image_identity(args.image)
    elif not workspace_path(args.binary).is_file():
        raise ValueError("Local Vision benchmark executable missing")
    folder = new_run(parent, "vision")
    metadata = {"generated_at": datetime.now(timezone.utc).isoformat(), "arguments": vars(args),
                "cases": cases, "models": {"baseline": {"path": str(baseline), "sha256": digest(baseline)}},
                "profile_enabled": args.profile,
                "accuracy_note": "Feature allclose is not detection/mask accuracy validation; no automatic promotion."}
    metadata["image_sha256"] = {path: digest(Path(path)) for path in sorted({c["image"] for c in cases})}
    if not args.local:
        metadata["docker_image"] = image_info
    if candidate:
        metadata["models"]["candidate"] = {"path": str(candidate), "sha256": digest(candidate)}
    write_json(folder / "inputs.json", metadata)
    manifest = folder / "cases.tsv"
    with manifest.open("w", encoding="utf-8", newline="\n") as stream:
        for case in cases:
            path = str(case["image"]) if args.local else container_path(Path(case["image"]))
            suffix = "\t" + "\t".join(map(str, case["roi"])) if case["roi"] is not None else ""
            stream.write(path + suffix + "\n")
    if candidate:
        needed = len(cases)*FEATURE_BYTES_PER_CASE + 512*1024*1024
        if shutil.disk_usage(folder).free < needed:
            raise RuntimeError(f"Feature comparison needs approximately {needed/1024**3:.2f} GiB free disk")
    print(f"Report directory: {folder}\nCases={len(cases)} iterations={args.iterations} repeats={args.repeats}", flush=True)
    if args.local:
        print("Local mode: manually ensure device is idle and logical device ID is correct.", flush=True)
    if args.profile:
        print("Profiling adds overhead: do not use this run as an unprofiled performance comparison.", flush=True)
    reports: dict[str, list[dict]] = {"baseline": [], "candidate": []}
    feature_result = None
    # Created by this run only; never delete a caller-supplied directory.
    feature_path = Path(tempfile.mkdtemp(prefix="feature-reference-", dir=folder))
    cleanup = contextlib.nullcontext() if args.keep_features else contextlib.ExitStack()
    with cleanup as stack:
        if not args.keep_features:
            stack.callback(shutil.rmtree, feature_path)
        for repeat in range(args.repeats):
            order = ["baseline", "candidate"] if candidate else ["baseline"]
            if repeat % 2:
                order.reverse()
            for label in order:
                output = folder / f"{label}-{repeat+1}.json"
                command = native_command(args, baseline if label == "baseline" else candidate, manifest, output,
                                         feature_path if candidate and repeat == 0 and label == "baseline" else None,
                                         feature_path if candidate and repeat == 0 and label == "candidate" else None)
                code = logged_run(command, output.with_suffix(".log"), accepted=(0, 2))
                data = json.loads(output.read_text(encoding="utf-8"))
                validate_report(data, cases, args)
                checks_passed = all(f["passed"] for c in data["cases"] for f in c["features"])
                if checks_passed != data.get("features_passed"):
                    raise ValueError("Native feature verdict inconsistent with per-output checks")
                if not data.get("features_passed", False) and not data.get("comparison_enabled", False):
                    raise RuntimeError(f"Non-finite model outputs: {output}")
                if code == 2 and not data.get("comparison_enabled", False):
                    raise RuntimeError(f"Invalid baseline/candidate outputs: {output}")
                reports[label].append(data)
                if data.get("comparison_enabled", False):
                    feature_result = {"passed": bool(data["features_passed"]), "atol": args.atol, "rtol": args.rtol,
                                      "cases": [{"image": c["image"], "roi": c["roi"], "features": c["features"]}
                                                for c in data["cases"]]}
    metadata["timings"] = {label: aggregate(data) for label, data in reports.items() if data}
    metadata["feature_comparison"] = feature_result
    for label, path in (("baseline", baseline), ("candidate", candidate)):
        if path is not None and digest(path) != metadata["models"][label]["sha256"]:
            raise RuntimeError("Model changed during benchmark; results cannot be compared")
    if any(digest(Path(path)) != sha for path, sha in metadata["image_sha256"].items()):
        raise RuntimeError("Images changed during benchmark; results cannot be compared")
    if candidate:
        if feature_result is None:
            raise RuntimeError("Candidate feature comparison missing")
        if any(r["soc"] != reports["baseline"][0]["soc"] for data in reports.values() for r in data):
            raise RuntimeError("Baseline/candidate were measured on different chip types")
        metadata["speed_comparison"] = speed_comparison(metadata["timings"]["baseline"], metadata["timings"]["candidate"])
    write_json(folder / "summary.json", metadata)
    for label, timing in metadata["timings"].items():
        for mode, stages in timing.items():
            t = stages["inference_ms"]
            print(f"{label:9s} {mode:4s}: inference mean={t['mean']:.3f}ms P50={t['p50']:.3f}ms P95={t['p95']:.3f}ms")
    if candidate:
        change = metadata["speed_comparison"]["all"]["inference_ms"]
        print(f"Vision inference latency reduction: {change['latency_reduction_percent']:.2f}%")
        print(f"Feature allclose: {'PASS' if feature_result['passed'] else 'FAIL'} (not a mask accuracy verdict)")
    print(f"Summary: {folder / 'summary.json'}")
    return 2 if feature_result and not feature_result["passed"] else 0


def main() -> int:
    try:
        args = parse_args()
        return run(args)
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        print(f"Vision benchmark error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
