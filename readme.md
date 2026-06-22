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

- **方式一**：传入 `input_ids` / `attention_mask`，由 `text-encoder.om` 推理得到 text feature。`sam3_demo` 内部使用 tokenizer 将原始文本转成 token。
- **方式二**：直接传入外部预计算的 `text_features` / `text_mask`，跳过 text encoder。

另外，`Sam3Input::need_mask` 用于控制是否输出分割掩码：

- `need_mask = true`（默认）：后处理会解码 `pred_masks`，每个结果可能包含 `segmentation`。
- `need_mask = false`：跳过 mask 解码，仅返回检测框，可减少 D2H 与 OpenCV 开销。

---

## 模型规格

### Vision Encoder

| 名称 | 形状 | 说明 |
|------|------|------|
| 输入 `images` | `[1, 3, 1008, 1008]` | float32，NCHW |
| 输出 `fpn_feat_0` | `[1, 256, 288, 288]` | 多尺度特征 0 |
| 输出 `fpn_feat_1` | `[1, 256, 144, 144]` | 多尺度特征 1 |
| 输出 `fpn_feat_2` | `[1, 256, 72, 72]` | 多尺度特征 2 |

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

1. **单 batch 推理**：`Sam3Infer` 缓存上一次输入的图像，图片未变化时不再重复执行 Vision Encoder。
2. **Vision 输出复用**：同一图片的多个 text prompt/word 推理时，Vision Encoder 的输出被反复复用。
3. **Text 输出常驻显存**：`TextModel` 的输出 buffer 在对象生命周期内始终位于 NPU 显存，不重复分配/释放。
4. **fpn_pos_2 常驻显存**：`fpn_pos_2_constant.npy` 在 `Sam3Infer::initialize()` 时加载并上传到 NPU，后续推理反复复用。
5. **Decoder 零拷贝输入**：`DecoderModel` 直接使用 VisionModel / TextModel / fpn_pos_2 的 device 输出指针构造输入 dataset，避免 D2D 拷贝。

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

### `sam3_demo`

```bash
./sam3_demo <vision_model> <text_model> <decoder_model> <fpn_pos2_npy> <tokenizer_json> <image_path> [prompt_text] [output_path]
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

### `sam3_bench`

```bash
./sam3_bench <vision_model> <text_model> <decoder_model> <fpn_pos2_npy> <tokenizer_json> <image_path> <prompt_text>
```

### 示例

```bash
./build/sam3_bench models/om-models/vision-encoder-tuned-2.om models/om-models/text-encoder.om models/om-models/decoder.om models/om-models/fpn_pos_2_constant.npy models/onnx-models/tokenizer.json workspace/persons.jpg "person" 

./build/sam3_demo models/om-models/vision-encoder-tuned-2.om models/om-models/text-encoder.om models/om-models/decoder.om models/om-models/fpn_pos_2_constant.npy models/onnx-models/tokenizer.json workspace/persons.jpg "person" workspace/result.jpg

```

---
## AOE 调优

### Vision Encoder

```bash
aoe --model=models/onnx-models/vision-encoder.onnx \
    --framework=5 \
    --output=models/om-models/vision-encoder-tuned-2 \
    --job_type=2 \
    --input_shape="images:1,3,1008,1008" \
    --insert_op_conf=models/config/vision.cfg
```

### Decoder

```bash
aoe --model=models/onnx-models/decoder_static.onnx \
    --framework=5 \
    --output=models/om-models/decoder_static-tuned \
    --job_type=2
```

---

## 模型转换

当前仓库中的 `models/om-models/*.om` 是通过 `soc_version=Ascend310` 转换的。若目标设备为 Ascend310P3，请使用对应 `soc_version` 重新转换。

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
    --input_format=ND
```

> 实际 `soc_version` 请与目标设备保持一致。
