"""Shared, standard-library-only helpers for offline Vision experiments."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTAINER_WORKSPACE = "/workspace"
DEFAULT_CANN = "swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-310p-ubuntu22.04-py3.11"


def settings(root: Path = ROOT) -> dict[str, str]:
    """Read literal .env values without executing shell code; caller env wins."""
    config: dict[str, str] = {}
    path = root / ".env"
    if path.is_file():
        for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:]
            key, separator, value = line.partition("=")
            key = key.strip()
            if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                raise ValueError(f"Invalid .env line {number}")
            parts = shlex.split(value, comments=True, posix=True)
            if len(parts) > 1:
                raise ValueError(f"Quote spaces in .env line {number}")
            config[key] = parts[0] if parts else ""
    config.update(os.environ)
    return config


def workspace_path(value: str | Path, root: Path = ROOT) -> Path:
    # Keep accepting service .env paths under /app, but experimental containers
    # mount project data elsewhere so the image's /app/bin stays visible.
    for prefix in (CONTAINER_WORKSPACE + "/", "/app/"):
        if str(value).startswith(prefix):
            value = str(value)[len(prefix):]
            break
    path = Path(value)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    path.relative_to(root.resolve())  # Docker only mounts the project; reject escaping symlinks.
    if any(c in str(path) for c in "\t\r\n"):
        raise ValueError("Paths must not contain tabs/newlines")
    return path


def container_path(path: Path, root: Path = ROOT) -> str:
    return CONTAINER_WORKSPACE + "/" + path.resolve().relative_to(root.resolve()).as_posix()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def new_run(parent: Path, prefix: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = parent / f"{prefix}-{stamp}-{uuid.uuid4().hex[:8]}"
    path.mkdir()
    return path


def write_json(path: Path, data: object) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def ensure_idle(device: int, docker: str = "docker", allow_privileged: bool = False) -> None:
    """Prevent a second container namespace opening a device used by a service."""
    ids = subprocess.check_output([docker, "ps", "-q"], text=True).split()
    if not ids:
        return
    containers = json.loads(subprocess.check_output([docker, "inspect", *ids], text=True))
    node = f"/dev/davinci{device}"
    busy = []
    unverified = []
    for item in containers:
        host = item.get("HostConfig", {})
        mapped = any(d.get("PathOnHost") == node for d in (host.get("Devices") or []))
        # Privileged containers can see every device; cannot prove this device idle.
        if mapped:
            busy.append(item.get("Name", item.get("Id", "unknown")).lstrip("/"))
        elif host.get("Privileged"):
            unverified.append(item.get("Name", item.get("Id", "unknown")).lstrip("/"))
    if busy:
        raise RuntimeError(f"Device {device} is accessible to running containers: {', '.join(busy)}. "
                           "Stop the relevant service before testing/tuning; do not scale another container on this device.")
    if unverified:
        message = "Privileged peer containers can access this device: " + ", ".join(unverified)
        if not allow_privileged:
            raise RuntimeError(message + ". Verify the selected device is idle with npu-smi, then explicitly "
                               "use --allow-privileged-peer if those containers only use OTHER devices.")
        print("WARNING: " + message + "; relying on your confirmation that they use only other devices.", flush=True)


def image_identity(image: str) -> dict:
    item = json.loads(subprocess.check_output(["docker", "image", "inspect", image], text=True))[0]
    return {"name": image, "id": item["Id"], "repo_digests": item.get("RepoDigests", []),
            "architecture": item.get("Architecture"), "os": item.get("Os")}


def docker_args(device: int, image: str, entrypoint: str, root: Path = ROOT,
                docker: str = "docker") -> list[str]:
    if device < 0:
        raise ValueError("Physical device must be nonnegative")
    nodes = [f"/dev/davinci{device}", "/dev/davinci_manager", "/dev/devmm_svm", "/dev/hisi_hdc"]
    for node in nodes:
        if not Path(node).exists():
            raise FileNotFoundError(f"NPU node missing: {node}")
    driver = Path("/usr/local/Ascend/driver")
    if not driver.is_dir():
        raise FileNotFoundError(f"Driver directory missing: {driver}")
    args = [docker, "run", "--rm", "--ipc=host", "-w", CONTAINER_WORKSPACE,
            "-e", "ASCEND_DEVICE_ID=0", "-v", f"{root.resolve()}:{CONTAINER_WORKSPACE}",
            "-v", f"{driver}:{driver}:ro"]
    for node in nodes:
        args.extend(["--device", f"{node}:{node}"])
    for path in (Path("/usr/local/Ascend/develop"), Path("/etc/ascend_install.info")):
        if path.exists():
            args.extend(["-v", f"{path}:{path}:ro"])
    args.extend(["--entrypoint", entrypoint, image])
    return args


def logged_run(command: list[str], log: Path, accepted: tuple[int, ...] = (0,)) -> int:
    """No shell interpolation; retain live progress plus an exact command record."""
    print("Running:", shlex.join(command), flush=True)
    with log.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(shlex.join(command) + "\n")
        stream.flush()
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace", bufsize=1) as process:
            assert process.stdout is not None
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    stream.write(line)
                    stream.flush()
                code = process.wait()
            except BaseException:
                process.terminate()
                raise
    if code not in accepted:
        raise RuntimeError(f"Command exited {code}; see {log}")
    return code
