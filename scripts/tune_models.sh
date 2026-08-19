#!/bin/bash
# 使用 Docker 运行 AOE 对 ONNX 模型进行自动调优，输出 tuned OM 模型
# 用法：
#   ./scripts/tune_models.sh
#   SOC_VERSION=Ascend310P3 ./scripts/tune_models.sh
#   FORCE=1 ./scripts/tune_models.sh   # 强制重新调优（覆盖已有输出）

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "${PROJECT_ROOT}/.env"
    set +a
fi
SOC_VERSION="${SOC_VERSION:-Ascend310P3}"
PHYSICAL_DEVICE_ID="${ASCEND_PHYSICAL_DEVICE_ID:-${DEVICE_ID:-2}}"
FORCE="${FORCE:-0}"

IMAGE="${CANN_IMAGE:-swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-310p-ubuntu22.04-py3.11}"
MODEL_DIR="/app/models"

echo "======================================"
echo "AOE 调优环境：Docker + ${IMAGE}"
echo "目标 SOC：${SOC_VERSION}"
echo "使用物理 Device ID：${PHYSICAL_DEVICE_ID}"
echo "项目目录：${PROJECT_ROOT}"
echo "======================================"

# 检查 onnx 模型是否存在
for model in vision-encoder decoder_static; do
    if [ ! -f "${PROJECT_ROOT}/models/onnx-models/${model}.onnx" ]; then
        echo "错误：models/onnx-models/${model}.onnx 不存在"
        exit 1
    fi
done

DEVICE_PATH="/dev/davinci${PHYSICAL_DEVICE_ID}"
if [ ! -e "${DEVICE_PATH}" ]; then
    echo "错误：${DEVICE_PATH} 不存在"
    exit 1
fi

DOCKER_ARGS=(
    --rm
    --device "${DEVICE_PATH}:${DEVICE_PATH}"
    --device /dev/davinci_manager:/dev/davinci_manager
    --device /dev/devmm_svm:/dev/devmm_svm
    --device /dev/hisi_hdc:/dev/hisi_hdc
    --ipc=host
    -e ASCEND_DEVICE_ID=0
    -v "${PROJECT_ROOT}:/app"
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro
)
if [ -d "/usr/local/Ascend/develop" ]; then
    DOCKER_ARGS+=(-v /usr/local/Ascend/develop:/usr/local/Ascend/develop:ro)
fi

run_aoe() {
    local name=$1
    local onnx_path=$2
    local output=$3
    shift 3
    local output_file="${PROJECT_ROOT}/models/om-models/${output}.om"

    if [ "${FORCE}" != "1" ] && [ -f "${output_file}" ]; then
        echo "[跳过] ${output_file} 已存在（设置 FORCE=1 可强制重新调优）"
        return 0
    fi

    echo ""
    echo "[AOE 调优] ${name} -> ${output}.om"
    docker run "${DOCKER_ARGS[@]}" \
        "${IMAGE}" \
        aoe --model="${MODEL_DIR}/onnx-models/${onnx_path}" \
            --framework=5 \
            --output="${MODEL_DIR}/om-models/${output}" \
            --job_type=2 \
            "$@"
}

# Vision Encoder
run_aoe "Vision Encoder" "vision-encoder.onnx" "vision-encoder-tuned" \
    --input_format=NCHW \
    --input_shape="images:1,3,1008,1008" \
    --insert_op_conf="${MODEL_DIR}/config/vision.cfg"

# Decoder
# run_aoe "Decoder" "decoder_static.onnx" "decoder_static-tuned" \
#     --input_shape="fpn_feat_0:1,256,288,288;fpn_feat_1:1,256,144,144;fpn_feat_2:1,256,72,72;fpn_pos_2:1,256,72,72;prompt_features:1,32,256;prompt_mask:1,32"
