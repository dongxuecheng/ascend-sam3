#!/bin/bash
# 使用 Docker 运行 AOE 对 ONNX 模型进行自动调优，输出 tuned OM 模型
# 用法：
#   ./scripts/tune_models.sh
#   SOC_VERSION=Ascend310P3 ./scripts/tune_models.sh
#   FORCE=1 ./scripts/tune_models.sh   # 强制重新调优（覆盖已有输出）

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOC_VERSION="${SOC_VERSION:-Ascend310P3}"
DEVICE_ID="${DEVICE_ID:-0}"
FORCE="${FORCE:-0}"

IMAGE="swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-310p-ubuntu22.04-py3.11"
MODEL_DIR="/app/models"

echo "======================================"
echo "AOE 调优环境：Docker + ${IMAGE}"
echo "目标 SOC：${SOC_VERSION}"
echo "使用 NPU ID：${DEVICE_ID}（映射由 ASCEND_VISIBLE_DEVICES 控制）"
echo "项目目录：${PROJECT_ROOT}"
echo "======================================"

# 检查 onnx 模型是否存在
for model in vision-encoder decoder_static; do
    if [ ! -f "${PROJECT_ROOT}/models/onnx-models/${model}.onnx" ]; then
        echo "错误：models/onnx-models/${model}.onnx 不存在"
        exit 1
    fi
done

# 挂载驱动目录（aoe 运行时需要 libascend_hal.so 等驱动库）
DRIVER_MOUNT=""
if [ -d "/usr/local/Ascend/driver" ]; then
    DRIVER_MOUNT="-v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro"
else
    echo "警告：未找到 /usr/local/Ascend/driver，调优可能因缺少驱动库失败"
fi

# 自动收集本机 Ascend 设备节点
DEVICE_MOUNT=""
for dev in /dev/davinci*; do
    [ -e "${dev}" ] && DEVICE_MOUNT="${DEVICE_MOUNT} --device ${dev}"
done
if [ -e "/dev/davinci_manager" ]; then
    DEVICE_MOUNT="${DEVICE_MOUNT} --device /dev/davinci_manager"
fi
if [ -e "/dev/devmm_svm" ]; then
    DEVICE_MOUNT="${DEVICE_MOUNT} --device /dev/devmm_svm"
fi
if [ -e "/dev/hisi_hdc" ]; then
    DEVICE_MOUNT="${DEVICE_MOUNT} --device /dev/hisi_hdc"
fi

run_aoe() {
    local name=$1
    local onnx_path=$2
    local output=$3
    shift 3
    local extra_args="$@"

    if [ "${FORCE}" != "1" ] && [ -f "${PROJECT_ROOT}/models/om-models/${output}.om" ]; then
        echo "[跳过] ${output}.om 已存在（设置 FORCE=1 可强制重新调优）"
        return 0
    fi

    echo ""
    echo "[AOE 调优] ${name} -> ${output}.om"
    docker run --rm \
        --privileged \
        ${DEVICE_MOUNT} \
        -e PATH="/usr/local/python3.11.15/bin:/usr/local/Ascend/ascend-toolkit/latest/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
        -e ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest \
        -e ASCEND_VISIBLE_DEVICES=${DEVICE_ID} \
        -e ASCEND_DEVICE_ID=0 \
        -e LD_LIBRARY_PATH="/usr/local/python3.11.15/lib:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/opskernel:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/develop/lib64" \
        -v "${PROJECT_ROOT}:/app" \
        ${DRIVER_MOUNT} \
        -itd \
        ${IMAGE} \
        bash -c "
            sed -i 's/int(np.random.choice(len(self.dims), size=1, p=point_probs))/np.random.choice(len(self.dims), p=point_probs)/' \
                /usr/local/Ascend/cann-9.0.0/python/site-packages/auto_tune/tuner_alg/ga_tuner.py && \
            aoe --model='${MODEL_DIR}/onnx-models/${onnx_path}' \
                --framework=5 \
                --output='${MODEL_DIR}/om-models/${output}' \
                --job_type=2 \
                ${extra_args}
        "
}

# Vision Encoder
run_aoe "Vision Encoder" "static.onnx" "vision-encoder-tuned-900-2" \
    --input_format=NCHW \
    --input_shape="images:1,3,1008,1008" \
    --insert_op_conf="${MODEL_DIR}/config/vision.cfg"

# Decoder
# run_aoe "Decoder" "decoder_static.onnx" "decoder_static-tuned" \
#     --input_shape="fpn_feat_0:1,256,288,288;fpn_feat_1:1,256,144,144;fpn_feat_2:1,256,72,72;fpn_pos_2:1,256,72,72;prompt_features:1,32,256;prompt_mask:1,32"