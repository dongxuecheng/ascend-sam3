#!/bin/bash
# 使用 Docker 运行 ATC 将 ONNX 模型转换为昇腾 OM 模型
# 用法：
#   ./scripts/convert_models.sh
#   SOC_VERSION=Ascend310P3 ./scripts/convert_models.sh
#   FORCE=1 ./scripts/convert_models.sh   # 强制重新转换（覆盖已有 .om）

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOC_VERSION="${SOC_VERSION:-Ascend310P3}"
FORCE="${FORCE:-0}"

IMAGE="swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.2.rc1-310p-ubuntu22.04-py3.11"
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

# 挂载驱动目录（atc 运行时需要 libascend_hal.so 等驱动库）
DRIVER_MOUNT=""
if [ -d "/usr/local/Ascend/driver" ]; then
    DRIVER_MOUNT="-v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro"
else
    echo "警告：未找到 /usr/local/Ascend/driver，转换可能因缺少驱动库失败"
fi

run_atc() {
    local name=$1
    local onnx_path=$2
    local output=$3
    shift 3
    local extra_args="$@"

    if [ "${FORCE}" != "1" ] && [ -f "${PROJECT_ROOT}/${output}.om" ]; then
        echo "[跳过] ${output}.om 已存在（设置 FORCE=1 可强制重新转换）"
        return 0
    fi

    echo ""
    echo "[转换] ${name} -> ${output}.om"
    docker run --rm \
        --privileged \
        -e PATH="/usr/local/python3.11.13/bin:/usr/local/Ascend/ascend-toolkit/latest/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
        -e ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest \
        -e LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/opskernel:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/develop/lib64 \
        -v "${PROJECT_ROOT}:/app" \
        ${DRIVER_MOUNT} \
        ${IMAGE} \
        atc --model="${MODEL_DIR}/onnx-models/${onnx_path}" \
            --framework=5 \
            --output="${MODEL_DIR}/om-models/${output}" \
            --soc_version="${SOC_VERSION}" \
            ${extra_args}
}

# Vision Encoder
run_atc "Vision Encoder" "vision-encoder.onnx" "vision-encoder" \
    --input_format=NCHW \
    --input_shape="images:1,3,1008,1008" \
    --insert_op_conf="${MODEL_DIR}/config/vision.cfg"

# Text Encoder
run_atc "Text Encoder" "text-encoder.onnx" "text-encoder" \
    --input_format=ND \
    --input_shape="input_ids:1,32;attention_mask:1,32"

# Decoder（onnx 中输入 shape 已固定，这里显式写出便于排查）
run_atc "Decoder" "decoder_static.onnx" "decoder_static" \
    --input_format=ND \
    --input_shape="fpn_feat_0:1,256,288,288;fpn_feat_1:1,256,144,144;fpn_feat_2:1,256,72,72;fpn_pos_2:1,256,72,72;prompt_features:1,32,256;prompt_mask:1,32"

echo ""
echo "======================================"
echo "模型转换完成，输出目录：${PROJECT_ROOT}/models/om-models/"
echo "======================================"
