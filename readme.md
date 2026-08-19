# Ascend SAM3 Inference

基于华为昇腾 CANN 原生 ACL 接口的 SAM3 端到端推理项目。

---

## 目录

- [项目简介](#项目简介)
- [模型规格](#模型规格)
- [核心设计](#核心设计)
- [后处理与 Mask 解码](#后处理与-mask-解码)
- [构建](#构建)
- [运行](#运行)
- [AOE 调优](#aoe-调优)
- [模型转换](#模型转换)

---

## 项目简介

本项目将 SAM3 拆分为三个 OM 模型在 Ascend NPU 上推理：

1. **Vision Encoder**：提取图像多尺度特征。
2. **Text Encoder**：将文本 prompt 编码为 text feature。
3. **Decoder**：融合 image feature 与 text feature，输出检测框、mask 和置信度。

推理入口为 `Sam3Infer`，`Sam3Input` 支持以下两种文本输入方式：

- **方式一**：传入 `input_ids` / `attention_mask`，由 `text-encoder.om` 推理得到 text feature。`ascendsam3_demo` 内部使用 tokenizer 将原始文本转成 token。
- **方式二**：直接传入外部预计算的 `text_features` / `text_mask`，跳过 text encoder。

另外，`Sam3Input::need_mask` 用于控制是否输出分割掩码：

- `need_mask = true`（默认）：后处理会解码 `pred_masks`，每个结果可能包含 `segmentation`。
- `need_mask = false`：跳过 mask 解码，仅返回检测框，可减少后处理开销。

---

## 模型规格

### Vision Encoder

| 名称 | 形状 | 说明 |
|------|------|------|
| 输入 `images` | `[1, 3, 1008, 1008]` | float32，NCHW |
| 输出 `fpn_feat_0` | `[1, 256, 288, 288]` | 多尺度特征 0 |
| 输出 `fpn_feat_1` | `[1, 256, 144, 144]` | 多尺度特征 1 |
| 输出 `fpn_feat_2` | `[1, 256, 72, 72]` | 多尺度特征 2 |
| 可选输出 `fpn_pos_2` | `[1, 256, 72, 72]` | 兼容部分四输出导出版本；运行时仍使用外部 `.npy` |

### Text Encoder

| 名称 | 形状 | 说明 |
|------|------|------|
| 输入 `input_ids` | `[1, 32]` | int64 |
| 输入 `attention_mask` | `[1, 32]` | int64 |
| 输出 `text_features` | `[1, 32, 256]` | float32 |
| 输出 `text_mask` | `[1, 32]` | 原始类型由模型决定 |

### Decoder

| 名称 | 形状 | 说明 |
|------|------|------|
| 输入 `fpn_feat_0/1/2` | 见 Vision Encoder | 来自 Vision Encoder |
| 输入 `fpn_pos_2` | `[1, 256, 72, 72]` | 来自 `fpn_pos_2_constant.npy` |
| 输入 `prompt_features` | `[1, 32, 256]` | 来自 Text Encoder |
| 输入 `prompt_mask` | `[1, 32]` | 来自 Text Encoder |
| 输出 `pred_masks` | `[1, 200, 288, 288]` | mask logit |
| 输出 `pred_boxes` | `[1, 200, 4]` | 归一化检测框 `[x1, y1, x2, y2]` |
| 输出 `pred_logits` | `[1, 200]` | 每 mask 的置信度 logit |
| 输出 `presence_logits` | `[1, 1]` | 图像整体存在性 logit |

---

## 核心设计

1. **单 batch 推理**：每个请求执行一次 Vision Encoder，同一请求中的多个文本类别复用该次图像特征。
2. **Vision 输出复用**：同一图片的多个 text prompt/word 推理时，Vision Encoder 的输出被反复复用。
3. **Text 输出常驻显存**：`TextModel` 的输出 buffer 在对象生命周期内始终位于 NPU 显存，不重复分配/释放。
4. **fpn_pos_2 常驻显存**：`fpn_pos_2_constant.npy` 在 `Sam3Infer::initialize()` 时加载并上传到 NPU，后续推理反复复用。
5. **Decoder 零拷贝输入**：`DecoderModel` 直接使用 VisionModel / TextModel / fpn_pos_2 的 device 输出指针构造输入 dataset，避免 D2D 拷贝。

---

## 后处理与 Mask 解码

### 后处理流程

`Sam3Infer::postprocess` 的执行步骤：

1. **CPU 筛选**：对 `pred_logits` / `presence_logits` 做 sigmoid，计算最终得分并排序，保留高于 `confidence_threshold` 的结果。
2. **Mask 解码**：当前稳定路径将选中的 `288x288` mask D2H，然后使用 OpenCV 按目标框裁剪、插值和二值化。
3. **CPU 裁剪与后处理**：按每张 mask 对应的检测框裁剪，并调用 `keep_largest_part()` 保留最大连通域。

`MaskPostprocessCann` 保留为后续优化入口，但当前版本未启用 aclnn 批量后处理。

### 开关：`Sam3Input::need_mask`

- `need_mask = true`（默认）：执行上述 mask 解码流程。
- `need_mask = false`：跳过所有 mask 解码，仅返回检测框，可减少 D2H 与后处理开销。

### 相关文件

- `src/infer/maskPostprocessCann.hpp` / `src/infer/maskPostprocessCann.cpp`
- `src/infer/sam3infer.cpp` 中的 `postprocess`

### 链接的 CANN 库

当前版本仅链接实际使用的 AscendCL/ACL Runtime 库。

---

## 构建

```bash
cd /home/HwHiAiUser/project/ascend-sam3
mkdir build && cd build
cmake ..
make -j4
```

### 依赖

- Ascend CANN Toolkit（需要 `ASCEND_HOME_PATH` 环境变量指向安装路径）
- OpenCV（必须通过环境变量 `OPENCV_INSTALL_DIR` 指定安装路径，路径下需包含 `include/opencv4`）
- C++17 编译器
- [tokenizers-cpp](https://github.com/mlc-ai/tokenizers-cpp)（已作为 `third_party/tokenizers-cpp` 子模块引入，首次构建会自动编译 Rust 绑定和 sentencepiece）
- Rust 工具链（构建 tokenizers-cpp 需要 `cargo` 和 `rustc`，可通过 [rustup](https://rustup.rs/) 安装）

---

## 运行

### `ascendsam3_demo`

```bash
./ascendsam3_demo <vision_model> <text_model> <decoder_model> <fpn_pos2_npy> <tokenizer_json> <image_path> [prompt_text] [output_path]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `vision_model` | vision encoder `.om` 文件路径 | - |
| `text_model` | text encoder `.om` 文件路径 | - |
| `decoder_model` | decoder `.om` 文件路径 | - |
| `fpn_pos2_npy` | `fpn_pos_2_constant.npy` 文件路径 | - |
| `tokenizer_json` | HuggingFace `tokenizer.json` 文件路径 | - |
| `image_path` | 输入图片路径 | - |
| `prompt_text` | 文本 prompt，例如 `"a person"` | 使用内置默认占位 prompt |
| `output_path` | 可视化结果保存路径 | `workspace/result.jpg` |

### `ascendsam3_bench`

```bash
./ascendsam3_bench <vision_model> <text_model> <decoder_model> <fpn_pos2_npy> <tokenizer_json> <image_path> <prompt_text>
```

### 示例

```bash
ASCEND_DEVICE_ID=2 ./build/ascendsam3_bench models/om-models/vision-encoder.om models/om-models/text-encoder.om models/om-models/decoder_static.om models/om-models/fpn_pos_2_constant.npy models/onnx-models/tokenizer.json workspace/persons.jpg "person"

ASCEND_DEVICE_ID=2 ./build/ascendsam3_demo models/om-models/vision-encoder.om models/om-models/text-encoder.om models/om-models/decoder_static.om models/om-models/fpn_pos_2_constant.npy models/onnx-models/tokenizer.json workspace/persons.jpg "person" workspace/result.jpg

```

---
## AOE 调优

### Vision Encoder

```bash
aoe --model=models/onnx-models/vision-encoder.onnx \
    --framework=5 \
    --output=models/om-models/vision-encoder-tuned \
    --job_type=2 \
    --input_shape="images:1,3,1008,1008" \
    --insert_op_conf=models/config/vision.cfg
```

---

## 模型转换

仓库不跟踪生成的 `.om` 文件。部署前必须按目标设备的实际 Chip Name 转换；Atlas 300I Duo 显示 `310P3` 时使用 `soc_version=Ascend310P3`。

提供了 Docker 一键转换脚本，无需在宿主机安装 CANN：

```bash
./scripts/convert_models.sh
```

运行所需 ONNX 为 `vision-encoder.onnx`、`text-encoder.onnx` 和 `decoder_static.onnx`；`geometry-encoder.onnx` 不在当前三模型推理链中使用。

支持通过环境变量覆盖配置：

```bash
# 指定目标芯片
SOC_VERSION=Ascend310P3 ./scripts/convert_models.sh

# 强制覆盖已存在的 .om
FORCE=1 ./scripts/convert_models.sh
```

### Vision Encoder

```bash
atc --model=models/onnx-models/vision-encoder.onnx \
    --framework=5 \
    --output=models/om-models/vision-encoder \
    --soc_version=Ascend310P3 \
    --input_format=NCHW \
    --input_shape="images:1,3,1008,1008" \
    --insert_op_conf=models/config/vision.cfg
```

### Text Encoder

```bash
atc --model=models/onnx-models/text-encoder.onnx \
    --framework=5 \
    --output=models/om-models/text-encoder \
    --soc_version=Ascend310P3 \
    --input_format=ND \
    --input_shape="input_ids:1,32;attention_mask:1,32"
```

### Decoder

```bash
atc --model=models/onnx-models/decoder_static.onnx \
    --framework=5 \
    --output=models/om-models/decoder_static \
    --soc_version=Ascend310P3 \
    --input_shape="fpn_feat_0:1,256,288,288;fpn_feat_1:1,256,144,144;fpn_feat_2:1,256,72,72;fpn_pos_2:1,256,72,72;prompt_features:1,32,256;prompt_mask:1,32"
```

> 实际 `soc_version` 请与目标设备保持一致。  
> 使用aoe对Vision Encoder模型进行优化，可以缩短时间，job_type=2时有效。     
> 使用aoe对Decoder模型进行优化后，几乎无收益，且模型结果错误。

---

## FastAPI 推理服务

项目已提供基于 `pybind11` 封装的 Python 模块 `ascendsam3` 与 FastAPI 服务 `service/main.py`。

### 接口

- `GET /health`：健康检查
- `POST /predict/file`：上传图片文件检测
- `POST /predict`：传入 base64 编码图片检测

### 请求参数

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `image` | file / string | 图片文件 或 base64 编码图片 | - |
| `class_names` | list[str] / str | 检测类别列表；file 接口用英文逗号分隔 | `["person"]` |
| `confidence` | float | 置信度阈值 | `0.3` |
| `return_mask` | bool | 是否返回 mask PNG | `true` |

### 本地启动

```bash
cd /home/HwHiAiUser/project/ascend-sam3
source /home/HwHiAiUser/Ascend/ascend-toolkit/set_env.sh
python3 -m uvicorn service.main:app --host 0.0.0.0 --port 8000
```

### Docker 部署

```bash
# 默认配置使用物理 device 2、容器内逻辑 device 0 和端口 18000，可在 .env 中覆盖。
cp .env.example .env

# 先转换出 vision-encoder.om、text-encoder.om、decoder_static.om
SOC_VERSION=Ascend310P3 ./scripts/convert_models.sh

# Docker 18.09 / Compose 1.22 可直接使用，不依赖 BuildKit。
docker-compose build
docker-compose config
docker-compose up -d --no-build
```

Dockerfile 将构建分成 `build-base`、`dependencies`、`builder` 和 `runtime`
四个阶段。首次构建仍需完整编译 Rust tokenizers、SentencePiece 和 Abseil；之后
仅修改 `src/` 时，`docker-compose build` 会复用 `dependencies` 阶段中的
`/app/build`，只重新编译 SAM3 C++/pybind 代码。修改 `third_party/`、依赖构建
配置或使用 `--no-cache` 才会重新编译第三方依赖。

模型目录通过 `docker-compose.yml` 只读挂载。容器不使用 `privileged`，默认只映射物理 `/dev/davinci2`；Docker 设备白名单将容器限制在该卡上，单卡可见后应用使用容器内索引 `ASCEND_DEVICE_ID=0`。

如需在空闲的物理 device 3 启动第二实例，请使用独立 Compose project、端口和镜像容器名，并设置 `ASCEND_PHYSICAL_DEVICE_ID=3`；单卡容器内仍使用 `ASCEND_LOGICAL_DEVICE_ID=0`。

### 调用示例

```bash
curl -X POST http://localhost:18000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "image": "<base64-image-string>",
    "class_names": ["person", "car"],
    "confidence": 0.3,
    "return_mask": true
  }'
```

