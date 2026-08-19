#!/bin/bash
# 使用 Docker 运行 ATC 将 ONNX 模型转换为昇腾 OM 模型
# 用法：
#   ./scripts/convert_models.sh
#   SOC_VERSION=Ascend310P3 ./scripts/convert_models.sh
#   FORCE=1 ./scripts/convert_models.sh   # 强制重新转换（覆盖已有 .om）

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "${PROJECT_ROOT}/.env"
    set +a
fi
SOC_VERSION="${SOC_VERSION:-Ascend310P3}"
FORCE="${FORCE:-0}"

IMAGE="${CANN_IMAGE:-swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-310p-ubuntu22.04-py3.11}"
MODEL_DIR="/app/models"

echo "======================================"
echo "模型转换环境：Docker + ${IMAGE}"
echo "目标 SOC：${SOC_VERSION}"
echo "项目目录：${PROJECT_ROOT}"
echo "======================================"

# 检查 onnx 模型是否存在
for model in vision-encoder text-encoder decoder_static; do
    if [ ! -f "${PROJECT_ROOT}/models/onnx-models/${model}.onnx" ]; then
        echo "错误：models/onnx-models/${model}.onnx 不存在"
        exit 1
    fi
done

mkdir -p "${PROJECT_ROOT}/models/om-models"

DOCKER_ARGS=(
    --rm
    -e ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
    -e LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/opskernel:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/develop/lib64
    -v "${PROJECT_ROOT}:/app"
)
if [ -d "/usr/local/Ascend/driver" ]; then
    DOCKER_ARGS+=(-v "/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro")
fi
if [ -d "/usr/local/Ascend/develop" ]; then
    DOCKER_ARGS+=(-v "/usr/local/Ascend/develop:/usr/local/Ascend/develop:ro")
fi

run_atc() {
    local name=$1
    local onnx_path=$2
    local output=$3
    shift 3
    local output_file="${PROJECT_ROOT}/models/om-models/${output}.om"

    if [ "${FORCE}" != "1" ] && [ -f "${output_file}" ]; then
        echo "[跳过] ${output_file} 已存在（设置 FORCE=1 可强制重新转换）"
        return 0
    fi

    echo ""
    echo "[转换] ${name} -> ${output}.om"
    docker run "${DOCKER_ARGS[@]}" \
        "${IMAGE}" \
        atc --model="${MODEL_DIR}/onnx-models/${onnx_path}" \
            --framework=5 \
            --output="${MODEL_DIR}/om-models/${output}" \
            --soc_version="${SOC_VERSION}" \
            "$@"
}

# 先转换体积较小且 I/O 约束最明确的 Decoder，尽早暴露输入名称/shape 问题。
run_atc "Decoder" "decoder_static.onnx" "decoder_static" \
    --input_format=ND \
    --input_shape="fpn_feat_0:1,256,288,288;fpn_feat_1:1,256,144,144;fpn_feat_2:1,256,72,72;fpn_pos_2:1,256,72,72;prompt_features:1,32,256;prompt_mask:1,32"

# Text Encoder
run_atc "Text Encoder" "text-encoder.onnx" "text-encoder" \
    --input_format=ND \
    --input_shape="input_ids:1,32;attention_mask:1,32"

# Vision Encoder（最后转换，静态 AIPP 将 BGR uint8 输入交换为 RGB 并归一化）
run_atc "Vision Encoder" "vision-encoder.onnx" "vision-encoder" \
    --input_format=NCHW \
    --input_shape="images:1,3,1008,1008" \
    --insert_op_conf="${MODEL_DIR}/config/vision.cfg"

echo ""
echo "======================================"
echo "模型转换完成，输出目录：${PROJECT_ROOT}/models/om-models/"
echo "======================================"
