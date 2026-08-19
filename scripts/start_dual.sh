#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
ENV_FILE=${SAM3_ENV_FILE:-"${PROJECT_DIR}/.env"}
COMPOSE_FILE="${PROJECT_DIR}/docker-compose.dual.yml"

if [[ -f "${ENV_FILE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
    set +a
fi

DEVICE_A_WORKERS=${SAM3_DEVICE_A_INSTANCES:-1}
DEVICE_B_WORKERS=${SAM3_DEVICE_B_INSTANCES:-1}
WORKER_HEALTHCHECK_TIMEOUT=${SAM3_WORKER_HEALTHCHECK_TIMEOUT:-180}

validate_worker_count() {
    local name=$1
    local value=$2

    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${name} must be a positive integer, got: ${value}" >&2
        exit 2
    fi

    # 防止配置错误瞬间启动过多模型进程。8不是硬件上限，只是部署保护值。
    if (( value > 8 )); then
        echo "${name} must not exceed 8, got: ${value}" >&2
        exit 2
    fi
}

validate_worker_count SAM3_DEVICE_A_INSTANCES "${DEVICE_A_WORKERS}"
validate_worker_count SAM3_DEVICE_B_INSTANCES "${DEVICE_B_WORKERS}"

if [[ ! "${WORKER_HEALTHCHECK_TIMEOUT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "SAM3_WORKER_HEALTHCHECK_TIMEOUT must be a positive integer, got: ${WORKER_HEALTHCHECK_TIMEOUT}" >&2
    exit 2
fi

echo "Starting SAM3 services:"
echo "  physical device ${SAM3_DEVICE_A:-2}: one container, ${DEVICE_A_WORKERS} worker(s)"
echo "  physical device ${SAM3_DEVICE_B:-3}: one container, ${DEVICE_B_WORKERS} worker(s)"
echo "  Uvicorn worker healthcheck timeout: ${WORKER_HEALTHCHECK_TIMEOUT}s"
echo "  gateway port: ${SAM3_PUBLIC_PORT:-18000}"

cd "${PROJECT_DIR}"

# 额外参数会原样传给 `docker-compose up`，例如：
#   bash scripts/start_dual.sh --force-recreate
#
# 两个 scale 值必须固定为 1。Ascend UDA 要求一个物理 Device 同时只能属于
# 一个容器 namespace；并发进程数由容器内的 SAM3_WORKERS 控制。
docker-compose -f "${COMPOSE_FILE}" up -d --no-build \
    "$@" \
    --scale "sam3-npu2=1" \
    --scale "sam3-npu3=1"

docker-compose -f "${COMPOSE_FILE}" ps
