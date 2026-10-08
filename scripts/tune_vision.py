#!/usr/bin/env python3
"""AOE Vision-only tuning with retained artifacts; never promotes the candidate."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from vision_workflow import (DEFAULT_CANN, container_path, digest, docker_args, image_identity,
                             ensure_idle, logged_run, new_run, settings, workspace_path, write_json)

# Runs INSIDE the same container as AOE. On 310P online tuning uses --device;
# --soc_version is an offline fast-tuning option for other product families.
PREFLIGHT = r'''
import ctypes, json, os, pathlib, subprocess, sys
os.environ.pop("ASCEND_RT_VISIBLE_DEVICES", None)
expected, folder, bank, *command = sys.argv[1:]
acl = ctypes.CDLL("libascendcl.so")
acl.aclInit.argtypes = [ctypes.c_char_p]
acl.aclrtGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
acl.aclrtSetDevice.argtypes = [ctypes.c_int32]
acl.aclrtGetSocName.restype = ctypes.c_char_p
def check(code, action):
    if code != 0: raise RuntimeError("%s failed: %d" % (action, code))
check(acl.aclInit(None), "aclInit")
bound = False
try:
    count = ctypes.c_uint32()
    check(acl.aclrtGetDeviceCount(ctypes.byref(count)), "aclrtGetDeviceCount")
    if count.value != 1: raise RuntimeError("Expected exactly one visible device, got %d" % count.value)
    check(acl.aclrtSetDevice(0), "aclrtSetDevice(0)")
    bound = True
    value = acl.aclrtGetSocName()
    if not value: raise RuntimeError("aclrtGetSocName returned null")
    actual = value.decode()
    if actual != expected: raise RuntimeError("Chip mismatch: expected %s, actual %s" % (expected, actual))
finally:
    if bound: check(acl.aclrtResetDevice(0), "aclrtResetDevice")
    check(acl.aclFinalize(), "aclFinalize")
help_result = subprocess.run(["aoe", "--help"], capture_output=True, text=True)
help_text = help_result.stdout + help_result.stderr
pathlib.Path(folder, "aoe-help.txt").write_text(help_text)
for option in ("--device", "--job_type", "--output", "--insert_op_conf", "--input_shape"):
    if option not in help_text: raise RuntimeError("AOE version missing required option: " + option)
pathlib.Path(bank).mkdir(parents=True, exist_ok=True)
os.environ["TUNE_BANK_PATH"] = bank
pathlib.Path(folder, "preflight.json").write_text(json.dumps({"soc": actual, "visible_devices": count.value,
    "logical_device": 0, "bank": bank, "command": command}, indent=2))
print("AOE preflight: actual_soc=%s device=0 bank=%s" % (actual, bank), flush=True)
os.chdir(folder)
os.execvp(command[0], command)
'''


def parse_args() -> argparse.Namespace:
    config = settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=config.get("ASCEND_PHYSICAL_DEVICE_ID", config.get("DEVICE_ID", "2")))
    parser.add_argument("--soc", default=config.get("SOC_VERSION", "Ascend310P3"), help="Verify actual chip; not an AOE compile flag")
    parser.add_argument("--image", default=config.get("CANN_IMAGE", DEFAULT_CANN))
    parser.add_argument("--onnx", default="models/onnx-models/vision-encoder.onnx")
    parser.add_argument("--aipp", default="models/config/vision.cfg")
    parser.add_argument("--output-name", default=config.get("VISION_TUNED_OUTPUT_NAME", "vision-encoder-trt-tuned"))
    parser.add_argument("--output-dir", default="benchmark-results")
    parser.add_argument("--force", action="store_true", default=config.get("FORCE", "0") == "1")
    parser.add_argument("--allow-privileged-peer", action="store_true", help="Confirm privileged peers use only other devices; never bypasses explicit same-device mapping")
    args = parser.parse_args()
    if args.device < 0:
        parser.error("Physical device must be nonnegative")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.output_name) or args.output_name.endswith(".om"):
        parser.error("Output name must not contain a path or .om suffix")
    if not args.soc.startswith("Ascend"):
        parser.error("Use a full chip name such as Ascend310P3")
    args.active_model = config.get("VISION_MODEL", "models/om-models/vision-encoder.om")
    return args


def make_command(args: argparse.Namespace, folder: Path, onnx: Path, aipp: Path) -> list[str]:
    prefix = container_path(folder / args.output_name)
    aoe = ["aoe", f"--model={container_path(onnx)}", "--framework=5", f"--output={prefix}",
           "--job_type=2", "--device=0", "--input_format=NCHW", "--input_shape=images:1,3,1008,1008",
           f"--insert_op_conf={container_path(aipp)}"]
    # Extend, rather than replace, the image's LD_LIBRARY_PATH (Python/TBE need
    # its original entries). Loading host driver libraries requires these paths.
    shell = 'unset ASCEND_RT_VISIBLE_DEVICES; export LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/develop/lib64:${LD_LIBRARY_PATH:-}"; exec "$@"'
    return docker_args(args.device, args.image, "/bin/bash") + ["-c", shell, "--", "python3", "-c", PREFLIGHT, args.soc,
        container_path(folder), container_path(folder / "bank"), *aoe]


def resolve_output(folder: Path, name: str) -> Path:
    exact = folder / (name + ".om")
    if exact.is_file() and exact.stat().st_size:
        return exact
    # Some versions append target OS/architecture. Never select arbitrary
    # intermediate OM files from aoe_workspace.
    matches = [p for p in folder.glob(name + "_*.om") if p.is_file() and p.stat().st_size]
    if len(matches) == 1:
        return matches[0]
    raise RuntimeError("AOE did not produce an unambiguous nonempty output OM at the requested path; "
                       f"artifacts retained in {folder}. Existing deployed models are unchanged.")


def publish(source: Path, target: Path, force: bool) -> None:
    if target.exists() and not force:
        raise FileExistsError(f"Candidate exists: {target}; choose another --output-name or explicitly use --force")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=target.stem + ".", suffix=".partial", dir=target.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        shutil.copyfile(source, temporary)
        if digest(source) != digest(temporary):
            raise RuntimeError("Candidate copy checksum mismatch")
        if target.exists() and not force:
            raise FileExistsError(f"Candidate appeared during tuning: {target}")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()  # Only this run's own mkstemp file.


def run(args: argparse.Namespace) -> int:
    onnx, aipp = workspace_path(args.onnx), workspace_path(args.aipp)
    target = workspace_path("models/om-models/" + args.output_name + ".om")
    active = workspace_path(args.active_model)
    if target == active:
        raise ValueError("Refusing to overwrite VISION_MODEL currently configured in .env/environment")
    if target.exists() and not args.force:
        raise FileExistsError(f"Candidate already exists: {target}. Use a new --output-name; no stale-output skip.")
    for path in (onnx, aipp):
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Input missing/empty: {path}")
    ensure_idle(args.device, allow_privileged=getattr(args, "allow_privileged_peer", False))
    image_info = image_identity(args.image)
    folder = new_run(workspace_path(args.output_dir), "aoe")
    metadata = {"generated_at": datetime.now(timezone.utc).isoformat(), "arguments": vars(args),
                "onnx_sha256": digest(onnx), "aipp_sha256": digest(aipp), "candidate": str(target),
                "status": "started", "auto_promoted": False, "docker_image": image_info}
    write_json(folder / "metadata.json", metadata)
    print(f"AOE artifacts: {folder}\nKeep the selected Device idle for the entire tuning run.", flush=True)
    try:
        command = make_command(args, folder, onnx, aipp)
        logged_run(command, folder / "aoe.log")
        source = resolve_output(folder, args.output_name)
        if digest(onnx) != metadata["onnx_sha256"] or digest(aipp) != metadata["aipp_sha256"]:
            raise RuntimeError("ONNX/AIPP changed during tuning; candidate was not published")
        preflight = json.loads((folder / "preflight.json").read_text(encoding="utf-8"))
        if preflight.get("soc") != args.soc:
            raise RuntimeError("Preflight chip validation missing or mismatched")
        publish(source, target, args.force)
        metadata.update(status="completed_unvalidated", output_sha256=digest(target),
                        output_bytes=target.stat().st_size, source=str(source), preflight=preflight,
                        bank_files=[p.relative_to(folder).as_posix() for p in (folder / "bank").rglob("*") if p.is_file()])
    except BaseException as error:
        metadata.update(status="failed", error=str(error))
        write_json(folder / "metadata.json", metadata)
        raise
    write_json(folder / "metadata.json", metadata)
    write_json(target.with_suffix(".build-info.json"), metadata)
    print(f"Candidate ready (NOT accepted/promoted): {target}\nBuild info: {target.with_suffix('.build-info.json')}")
    print("Next: compare with scripts/benchmark_vision.py, then validate boxes/masks through the service.")
    return 0


def main() -> int:
    try:
        args = parse_args()
        return run(args)
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        print(f"AOE error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
