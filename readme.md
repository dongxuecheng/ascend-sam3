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
- [FastAPI 推理服务](#fastapi-推理服务)
- [Docker 启动前准备](#docker-启动前准备)
- [Docker 单实例部署](#docker-单实例部署)
- [Docker 多实例统一入口](#docker-多实例统一入口)
- [日常启停与模式切换](#日常启停与模式切换)
- [本机调用与远程访问](#本机调用与远程访问)
- [多实例性能测试](#多实例性能测试)

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
- `GET /ui-config`：前端支持的模式和目标细化裁剪预算（不执行推理）
- `POST /predict/file`：上传图片文件检测
- `POST /predict`：传入 base64 编码图片检测
- `POST /predict-obj-refine`、`POST /predict-obj-refine/file`：主体预检测与局部精细检测

### 前端界面

访问 `http://服务器IP:18000/`。页面布局与
[trt-sam3 前端](https://github.com/leon0514/trt-sam3/tree/593b4ae199c23e6a2042a7dff268c79a9904b253/workspace/app/frontend)
对齐：深色三栏、左侧参数设置、中间图片和叠加结果、右侧目标列表。
支持拖拽上传、标签管理、Mask/框/文字开关、类别/置信度/面积筛选、
目标勾选与排序、JSON 查看及结果图下载；小屏幕自动调整布局。

推理请求仍使用昇腾接口，不需要 TensorRT 或新的前端依赖：

- 多类别文本提示：调用 `/predict/file`。
- 目标细化（obj-refine）：调用 `/predict-obj-refine/file`，可设置预检测标签、
  精细检测标签、合并原图结果及 Omnicrop 参数。精细标签应不同于预检测标签，
  例如预检测 `person`、精细检测 `helmet`。`max_crops` 留空使用服务器默认值，
  上限由 `.env` 的 `SAM3_REFINE_MAX_CROPS` 决定；触及预算会显示提示。
- 纯框、文本+框、跨图参考提示：保留模式提示但禁用，当前 Ascend 模型链路不支持。

前端资源位于 `service/static/`，随服务镜像一起构建。更新代码后选择当前部署模式
重建镜像；仅 `up --no-build` 不会把新页面复制到旧镜像：

```bash
# 单实例
docker-compose -f docker-compose.yml build sam3-service
docker-compose -f docker-compose.yml up -d --no-build --force-recreate

# 或多实例统一入口（不要与单实例同时启动）
docker-compose -f docker-compose.dual.yml build sam3-npu2
docker-compose -f docker-compose.dual.yml up -d --no-build --force-recreate
```

重建后浏览器使用 `Ctrl+F5` 强制刷新。几何提示不会因换界面自动获得支持。
前端逻辑测试（Node.js 20+，不需要 NPU）：

```bash
node --test tests/frontend.test.cjs
```

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

### Docker 启动前准备

以下命令在 Linux Ascend 服务器的项目目录执行，示例路径为 `/root/ascend-sam3`。
统一使用 `docker-compose`；安装 Compose V2 的服务器可将其替换为 `docker compose`，
其余参数不变。不需要额外启动脚本。默认配置下单实例和多实例共享 device 2 及
18000 端口，只选择一种部署方式；切换步骤见下文。

```bash
cd /root/ascend-sam3
# 保留已有 .env，不覆盖设备号、端口和 worker 配置。
cp -n .env.example .env
npu-smi info
```

先确认所选 Device 未被其他服务占用，且下列运行时模型文件均已准备好：

```text
models/om-models/vision-encoder.om
models/om-models/text-encoder.om
models/om-models/decoder_static.om
models/om-models/fpn_pos_2_constant.npy
models/onnx-models/tokenizer.json
```

如果尚未生成 OM，按[模型转换](#模型转换)准备 ONNX 后执行；已有可用 OM 可跳过：

```bash
SOC_VERSION=Ascend310P3 bash scripts/convert_models.sh
```

以下健康检查和调用示例使用默认端口；若修改 `.env`，请同步替换命令中的端口。

### Docker 单实例部署

使用 `docker-compose.yml`，启动一个 `sam3-service` 容器、一个 Uvicorn worker，
直接将宿主机 18000 映射到容器 8000，无 Nginx。检查 `.env`：

```dotenv
ASCEND_PHYSICAL_DEVICE_ID=2
ASCEND_LOGICAL_DEVICE_ID=0
SAM3_PORT=18000
```

`ASCEND_PHYSICAL_DEVICE_ID` 表示宿主机 `/dev/davinciN` 的设备编号；仅映射一颗
设备后，容器内使用逻辑编号 0。多实例的 `SAM3_DEVICE_*_INSTANCES` 不控制该模式。

首次部署或修改需要打包到镜像的代码时构建：

```bash
cd /root/ascend-sam3
docker-compose -f docker-compose.yml build sam3-service
```

已有 `ascend-sam3-service:latest` 镜像时，直接检查配置并启动：

```bash
cd /root/ascend-sam3
docker-compose -f docker-compose.yml config
docker-compose -f docker-compose.yml up -d --no-build
docker-compose -f docker-compose.yml ps
docker-compose -f docker-compose.yml logs --tail=100 sam3-service

# 日志出现 Application startup complete 后验证。
curl --max-time 10 -fsS http://127.0.0.1:18000/health
```

如果修改了 `.env`、设备映射或更新镜像，重建容器以应用配置：

```bash
docker-compose -f docker-compose.yml up -d --no-build --force-recreate
```

Dockerfile 将构建分成 `build-base`、`dependencies`、`builder` 和 `runtime`
四个阶段。首次构建仍需完整编译 Rust tokenizers、SentencePiece 和 Abseil；之后
仅修改 `src/` 时，`docker-compose build` 会复用 `dependencies` 阶段中的
`/app/build`，只重新编译 SAM3 C++/pybind 代码。修改 `third_party/`、依赖构建
配置或使用 `--no-cache` 才会重新编译第三方依赖。

模型目录通过 `docker-compose.yml` 只读挂载。容器不使用 `privileged`，默认只映射物理 `/dev/davinci2`；Docker 设备白名单将容器限制在该卡上，单卡可见后应用使用容器内索引 `ASCEND_DEVICE_ID=0`。

单实例配置仍使用 Compose 默认 bridge 网络，依赖 Docker 的网络状态和端口转发。
此前针对 `network ... not found` 的 host 网络规避方案只应用于下面的多实例配置，
不要将两种模式的网络检查命令混用。

### Docker 多实例统一入口

`docker-compose.dual.yml` 将 `sam3-npu2`、`sam3-npu3` 分别绑定到宿主机
device 2、device 3。每个 Device 固定只创建一个容器，容器内部由 Uvicorn
启动一个或多个独立 SAM3 worker 进程。Nginx 使用 `least_conn` 在两个 Device
容器之间分流，容器内再由 Uvicorn 将连接交给 worker。三个容器使用 Linux host
网络；两个后端分别只监听宿主机的 `127.0.0.1:18001`、`127.0.0.1:18002`，外部
客户端始终访问网关的一个端口，接口路径、参数和返回值与单实例完全相同。

使用 host 网络是针对本服务器 Docker 启动时网络状态库被清理的兼容措施。故障
日志包含 `cleanup DB .../network/files/local-kv.db`，自定义 bridge 网络消失，
旧容器因引用不存在的网络 ID 而启动失败。新配置不再引用该自定义网络；自动恢复
仍要求 Docker、Ascend 驱动和模型挂载正常，且容器未被人工停止或删除。
后端只绑定回环地址，不会将 18001、18002 暴露到外部网卡。

从旧 bridge 版本升级或已经出现 `network ... not found` 时，先移除持有旧网络 ID
的三个容器，再按新配置创建一次。该操作不会删除镜像和只读挂载的模型目录：

```bash
cd /root/ascend-sam3
docker-compose -f docker-compose.dual.yml down
docker-compose -f docker-compose.dual.yml \
  up -d --no-build --force-recreate
```

需要正常开机自动恢复时，保留三个服务的 `restart: unless-stopped`，不要在关机前
手工停止或删除容器。可以在维护窗口重启服务器后验证：

```bash
cd /root/ascend-sam3
docker-compose -f docker-compose.dual.yml ps
curl -fsS http://127.0.0.1:18001/health
curl -fsS http://127.0.0.1:18002/health
curl -fsS http://127.0.0.1:18000/health
```

Ascend UDA 驱动不允许两个不同的容器 namespace 同时打开同一个物理 Device；
这种配置会在内核日志中出现 `Conflict open udevid`，并使第二个容器的
`aclInit()` 返回 `500000`。`docker-compose.dual.yml` 中每个后端都是一个独立
Compose 服务，默认各创建一个容器；`.env` 中的实例数只控制容器内部 worker，
不能用于扩展共享同一 Device 的容器。不要对 `sam3-npu2`、`sam3-npu3` 使用
`docker-compose up --scale`。

每个后端容器只映射一个 `/dev/davinciN`，容器内可见设备会重新编号为 0，
所以所有实例均使用 `ASCEND_DEVICE_ID=0`。不要增加
`ASCEND_RT_VISIBLE_DEVICES`，否则会同时启用另一套设备过滤逻辑。

首次切换前检查 `.env`。已有 `.env` 不需要覆盖；以下变量即使不存在也会使用
Compose 文件中的默认值：

```bash
cd /root/ascend-sam3
cp -n .env.example .env

# 旧版 .env 没有实例数字段时补上默认值。
grep -q '^SAM3_DEVICE_A_INSTANCES=' .env || echo 'SAM3_DEVICE_A_INSTANCES=1' >> .env
grep -q '^SAM3_DEVICE_B_INSTANCES=' .env || echo 'SAM3_DEVICE_B_INSTANCES=1' >> .env
grep -q '^SAM3_WORKER_HEALTHCHECK_TIMEOUT=' .env || echo 'SAM3_WORKER_HEALTHCHECK_TIMEOUT=180' >> .env
grep -q '^SAM3_BACKEND_A_PORT=' .env || echo 'SAM3_BACKEND_A_PORT=18001' >> .env
grep -q '^SAM3_BACKEND_B_PORT=' .env || echo 'SAM3_BACKEND_B_PORT=18002' >> .env

grep -E 'SAM3_DEVICE_A|SAM3_DEVICE_B|SAM3_DEVICE_A_INSTANCES|SAM3_DEVICE_B_INSTANCES|SAM3_WORKER_HEALTHCHECK_TIMEOUT|SAM3_BACKEND_[AB]_PORT|SAM3_PUBLIC_PORT|SAM3_GATEWAY_IMAGE' .env || true
```

默认值如下，可根据实际设备号和镜像仓库修改：

```dotenv
SAM3_DEVICE_A=2
SAM3_DEVICE_B=3
SAM3_DEVICE_A_INSTANCES=1
SAM3_DEVICE_B_INSTANCES=1
SAM3_WORKER_HEALTHCHECK_TIMEOUT=180
SAM3_BACKEND_A_PORT=18001
SAM3_BACKEND_B_PORT=18002
SAM3_PUBLIC_PORT=18000
SAM3_GATEWAY_IMAGE=nginx:1.30.4-alpine
```

三个端口必须互不相同，并且不能与宿主机现有服务冲突。修改后端端口不会改变
客户端地址；业务始终访问 `SAM3_PUBLIC_PORT`。

外部访问还需要宿主机入站规则允许 `SAM3_PUBLIC_PORT`，见[本机调用与远程访问](#本机调用与远程访问)。

`SAM3_DEVICE_A_INSTANCES`、`SAM3_DEVICE_B_INSTANCES` 分别控制两个 Device
容器内部的 Uvicorn/SAM3 worker 数。每个 worker 都会独立执行 `aclInit()`、加载
一套 OM 模型并占用独立 NPU 内存。建议先使用 `1/1` 建立基线，再改为 `2/2`
测试。实例数必须是正整数；根据当前压测结果，建议使用 `1` 或 `2`，不要仅因
NPU 内存尚有空余就继续增加 worker。

SAM3 worker 会在 FastAPI startup 阶段同步加载三套 OM 模型。实测单 worker
初始化约需 28 秒，超过 Uvicorn 多进程管理器默认 5 秒的 worker 健康检查时间。
`SAM3_WORKER_HEALTHCHECK_TIMEOUT` 会传给
`uvicorn --timeout-worker-healthcheck`，默认设为 180 秒，以便两个 worker 并发
加载时留出余量。该值必须是正整数；如果服务器上的实测初始化时间明显更长，可
继续调大。

首次部署或更新 SAM3 代码时，只构建一次共享镜像；已有可用镜像可跳过构建：

```bash
docker-compose -f docker-compose.dual.yml build sam3-npu2
```

`sam3-npu3` 直接复用 `sam3-npu2` 构建产生的
`ascend-sam3-service:latest`。网关使用支持 aarch64 的 Nginx 官方镜像；如果
服务器无法访问 Docker Hub，可先把该镜像同步到内部仓库，再通过
`SAM3_GATEWAY_IMAGE` 指定完整镜像地址。

已有镜像时，日常启动直接执行下列命令。若单实例仍在运行，先按
[日常启停与模式切换](#日常启停与模式切换)停止它，以释放 Device 和端口：

```bash
cd /root/ascend-sam3
docker-compose -f docker-compose.dual.yml config
docker-compose -f docker-compose.dual.yml up -d --no-build
docker-compose -f docker-compose.dual.yml ps
```

镜像已存在时，`docker-compose -f docker-compose.dual.yml up -d` 同样可用；
加 `--no-build` 表示本次只启动，不自动构建。修改 `.env` 中的 worker 数或端口后，
执行 `up -d --no-build --force-recreate`，普通 `restart` 不会重新读取这些配置。

`docker-compose.dual.yml` 不再创建 `br-sam3`，也不再依赖 Docker DNS、DNAT 或
firewalld 的容器转发规则。Compose 的 `PORTS` 列在 host 网络模式下为空属于正常
现象；实际监听端口用 `ss -lntp` 检查。

如果服务器已经运行过旧版“同一 Device 扩展多个容器”的配置，必须先执行
`down`，让健康容器和反复重启的冲突容器全部释放 Device，再迁移到单容器多
worker。由于 worker PID 响应头来自更新后的 Python 服务，本次需要重新构建一次
SAM3 镜像；第三方依赖层没有变化时会直接复用构建缓存：

```bash
docker-compose -f docker-compose.dual.yml build sam3-npu2
docker-compose -f docker-compose.dual.yml down
docker-compose -f docker-compose.dual.yml \
  up -d --no-build --force-recreate
```

检查网关、后端和 NPU：

```bash
docker-compose -f docker-compose.dual.yml config \
  | grep -E 'network_mode|SAM3_(PUBLIC|BACKEND_[AB])_PORT'

ss -lntp | grep -E ':(18000|18001|18002)\b'

curl -fsS http://127.0.0.1:18000/gateway-health
curl --max-time 10 -fsS http://127.0.0.1:18001/health
curl --max-time 10 -fsS http://127.0.0.1:18002/health
curl --max-time 10 -fsS http://127.0.0.1:18000/health

docker logs --tail=100 sam3-gateway
docker-compose -f docker-compose.dual.yml logs --tail=100 sam3-npu2 sam3-npu3
npu-smi info
```

模型加载期间可能暂时连接失败或返回 502，待两个后端日志出现
`Application startup complete` 后重新检查。`/gateway-health` 只证明 Nginx 就绪；
网关 `/health` 成功只证明至少一个后端可用，因此应分别检查 18001 和 18002。

Compose 会自动生成带序号的后端容器名。下面两个结果都必须是 `1`；大于 1
说明仍残留旧版的同 Device 多容器配置：

```bash
docker-compose -f docker-compose.dual.yml ps -q sam3-npu2 | wc -l
docker-compose -f docker-compose.dual.yml ps -q sam3-npu3 | wc -l
```

检查容器内进程以及各 worker 的模型加载日志：

```bash
docker top $(docker-compose -f docker-compose.dual.yml ps -q sam3-npu2)
docker top $(docker-compose -f docker-compose.dual.yml ps -q sam3-npu3)

docker-compose -f docker-compose.dual.yml logs --tail=200 sam3-npu2 sam3-npu3 \
  | grep -E 'Started parent process|Started server process|SAM3 worker ready|aclInit failed'
```

每个成功 worker 都会输出一条 `SAM3 worker ready`。`npu-smi info` 中 device 2、
device 3 下的 SAM3 `python3` 进程数也应分别等于 `.env` 中配置的数量。

首页只注册了 `GET /`，所以 `curl -I` 发送 `HEAD /` 时返回 405 属于正常现象。
应使用 GET 检查首页：

```bash
curl -sS -D - -o /dev/null http://127.0.0.1:18000/
```

网关会在响应中增加 `X-SAM3-Upstream`，Python 服务增加
`X-SAM3-Worker-PID`。前者通过本机后端端口区分 Device 容器，后者用于区分
容器内 worker；
业务客户端不需要依赖这些运维响应头：

```bash
for i in 1 2 3 4; do
  curl -sS -D - -o /dev/null http://127.0.0.1:18000/health \
    | grep -Ei '^X-SAM3-(Upstream|Worker-PID):'
done
```

两个 Device 容器分别设置了健康检查和自动重启，Uvicorn 主进程负责管理和
重启容器内 worker。Nginx 对连接失败、超时以及
502/503/504 最多尝试两个后端；检测 POST 请求没有写入副作用，因此允许故障
重试。Nginx 的 upstream 固定指向两个仅回环可见的后端端口；端口通过 `.env`
传给 Nginx 模板，容器重建后不需要手工修改配置文件。

需要更新 SAM3 镜像时，只构建一次，然后按 `.env` 的 worker 数重建两个后端
容器和网关：

```bash
docker-compose -f docker-compose.dual.yml build sam3-npu2
docker-compose -f docker-compose.dual.yml \
  up -d --no-build --force-recreate
```

多实例提高的是并发吞吐和排队延迟。单个请求仍完整地交给其中一个实例处理；
只有同时存在多个请求时，各进程才会并行执行推理。相同 device 上的多个进程会
竞争同一组 AI Core 和内存带宽，因此显存能容纳不等于吞吐一定线性增长，必须以
下面的压测结果为准。

### 日常启停与模式切换

停止后再次启动（不删除容器），只执行当前部署模式对应的一组命令。
单实例：

```bash
# 按需执行 stop 或 up。
docker-compose -f docker-compose.yml stop
docker-compose -f docker-compose.yml up -d --no-build
```

多实例：

```bash
# 按需执行 stop 或 up。
docker-compose -f docker-compose.dual.yml stop
docker-compose -f docker-compose.dual.yml up -d --no-build
```

以下切换命令会中断 SAM3 服务并删除原模式的容器；镜像与宿主机 `models` 目录
保留。两种模式共用镜像，镜像已构建时无需再次构建。不要添加 `--remove-orphans`，
以免删除同一 Compose 项目下由其他配置管理的流水线容器。

```bash
cd /root/ascend-sam3
# 单实例切换到多实例。
docker-compose -f docker-compose.yml down
docker-compose -f docker-compose.dual.yml up -d --no-build
```

```bash
cd /root/ascend-sam3
# 多实例切换回单实例。
docker-compose -f docker-compose.dual.yml down
docker-compose -f docker-compose.yml up -d --no-build
```

两个配置均使用 `restart: unless-stopped`。正常重启服务器前无需执行 `down`；
手工 `stop` 的容器不会自动恢复，`down` 删除的容器需要重新执行对应的 `up`。
Docker 开机自启、设备节点和模型目录就绪是恢复的前提，健康检查本身不会让
一个仍在运行但 `unhealthy` 的容器自动重启。

### 本机调用与远程访问

两种模式的默认业务入口均为 `http://127.0.0.1:18000`。在服务器上先验证
`/health`，再用实际图片验证预测；将图片路径替换为已存在的文件：

```bash
curl --max-time 10 -fsS http://127.0.0.1:18000/health
curl --max-time 120 -fsS http://127.0.0.1:18000/predict/file \
  -F 'image=@test-images/example.jpg' \
  -F 'class_names=person' \
  -F 'confidence=0.3' \
  -F 'return_mask=false'
```

上传字段名是 `image`，不是 `file`。浏览器使用 `http://服务器IP:18000/`；
当前网关未配置 HTTPS。远程无法访问但本机预测成功时，不影响宿主机程序通过
`127.0.0.1:18000` 调用。host 网络容器也可使用此地址；普通 bridge 容器中的
`127.0.0.1` 指向容器自身，调用方需要使用可达的宿主机地址。

多实例的 host 网络共享宿主机网络栈，外部请求受宿主机 `INPUT` 及上游 ACL
控制。本机访问成功而远程失败时，检查 `ss -lntp` 的监听地址、防火墙、路由和
客户端代理，不要仅凭此现象重建 SAM3 镜像。若使用 firewalld，先用
`firewall-cmd --get-active-zones` 确认入口网卡所属区域。以下仅以 `public` 区域
及默认端口为例，按实际区域和访问策略放行：

```bash
firewall-cmd --permanent --zone=public --add-port=18000/tcp
firewall-cmd --zone=public --add-port=18000/tcp
```

只供本机调用时无需为外部访问额外放行；后端 18001、18002 只监听回环地址。
`firewalld` 未运行时检查现有 iptables/nftables 或上游规则，不要清空防火墙。

### 多实例性能测试

`scripts/benchmark_service.py` 使用 Python 标准库并发上传 `test-images` 目录下
所有 jpg/jpeg/png/bmp/webp 图片，输出总耗时、吞吐、平均/p50/p95/p99 延迟和
Nginx 实际 Device 容器及 worker PID 分布。预热请求不计入结果，图片会在计时
前读入内存，因此统计值主要反映 HTTP 和推理耗时。

先测试每个 device 一个实例：

```bash
cd /root/ascend-sam3

# .env
sed -i 's/^SAM3_DEVICE_A_INSTANCES=.*/SAM3_DEVICE_A_INSTANCES=1/' .env
sed -i 's/^SAM3_DEVICE_B_INSTANCES=.*/SAM3_DEVICE_B_INSTANCES=1/' .env
docker-compose -f docker-compose.dual.yml \
  up -d --no-build --force-recreate

until curl -fsS http://127.0.0.1:18000/health; do sleep 2; done
python3 scripts/benchmark_service.py \
  --images test-images \
  --concurrency 2 \
  --rounds 3 \
  --warmup 2 \
  --label 1-instance-per-device \
  --json-output benchmark-1x.json
```

再测试每个 device 两个实例：

```bash
sed -i 's/^SAM3_DEVICE_A_INSTANCES=.*/SAM3_DEVICE_A_INSTANCES=2/' .env
sed -i 's/^SAM3_DEVICE_B_INSTANCES=.*/SAM3_DEVICE_B_INSTANCES=2/' .env
docker-compose -f docker-compose.dual.yml \
  up -d --no-build --force-recreate

until curl -fsS http://127.0.0.1:18000/health; do sleep 2; done
python3 scripts/benchmark_service.py \
  --images test-images \
  --concurrency 4 \
  --rounds 3 \
  --warmup 4 \
  --label 2-instances-per-device \
  --json-output benchmark-2x.json
```

默认检测类别为 `person`、`fire`，默认不返回 mask，以免大响应体干扰推理吞吐。
可重复传入 `--class-name` 改类别；如需模拟实际返回 mask 的业务请求，再增加
`--return-mask`：

```bash
python3 scripts/benchmark_service.py \
  --images test-images \
  --concurrency 4 \
  --rounds 3 \
  --class-name person \
  --class-name fire \
  --return-mask
```

比较两个 JSON/终端结果时，优先看 `throughput_images_per_second`、p95 延迟、
失败数、`upstream_counts` 和 `worker_counts`。`upstream_counts` 在两种配置下
通常都只有两个本机后端地址（默认 `127.0.0.1:18001` 和 `127.0.0.1:18002`）；
配置为 `2/2` 时，`worker_counts` 应出现四个“后端地址 + PID”组合，表明四个
模型进程均收到请求。用于观察全部 worker 的压测并发数应至少等于总 worker 数；如果
图片只有十几张，使用 `--rounds 3` 或更高可降低偶然波动。压测期间同时执行
`watch -n 1 npu-smi info`，确认没有显存耗尽、温度降频或实例启动失败。建议
每组测试重复三次并取中位数。

### 调用示例

#### 主体预检测与局部精细检测（obj-refine）

新增 `POST /predict-obj-refine`（JSON）和
`POST /predict-obj-refine/file`（multipart）。现有 `/predict` 和 `/predict/file`
的参数与返回格式保持不变。新流程复用原有三个 OM、位置编码和 tokenizer，
不需要 geometry encoder，也不需要重新执行 ATC。

处理顺序：原图预检测 → 可选的原图精细检测（复用视觉特征）→ 主体框聚合裁剪
→ 每个裁剪区域精细检测 → 恢复原图坐标 → 按类别进行 IoU=0.5 的 NMS。
每个裁剪区域内的多个文本提示共用一次视觉编码，文本特征沿用现有缓存。
裁剪使用 CPU/OpenCV，模型推理仍走 ACL/AIPP；没有引入 CUDA/TensorRT。
裁剪聚合直接使用与 TRT 同源的 [OmniCrop C++ 核心](https://github.com/leon0514/OmniCrop)，
固定版本 `e9faf56f45df2ee69d028dfaa91b17b6b806b8dd`，位于
`src/third_party/omnicrop/OmniCrop.hpp`。输入校验和预算限制由 Ascend 适配层执行；
未触及预算时，相同且同序的预检测框、图像尺寸、裁剪参数会得到同一算法的浮点裁剪框。
来源与上游 MIT 许可证声明记录在同目录 `NOTICE.md`，也随运行镜像分发。
不额外安装 Python omnicrop 包，不修改现有 OM；Mask 解码、图像采样和模型数值
仍使用当前 Ascend 路径，因此不能保证最终检测/分割结果与 TRT 完全一致。

该头文件随 `src/` 在业务代码阶段编译，不会因接入 OmniCrop 而使
tokenizers/SentencePiece/Abseil 的独立依赖构建缓存失效。

**更新后必须构建新镜像**（仅重启旧镜像不会增加接口）：

```bash
# 单实例：选择这一组
docker-compose -f docker-compose.yml build sam3-service
docker-compose -f docker-compose.yml up -d --no-build --force-recreate

# 多实例：选择这一组，不要与单实例同时占用同一 Device
docker-compose -f docker-compose.dual.yml build sam3-npu2
docker-compose -f docker-compose.dual.yml up -d --no-build --force-recreate
```

也可以将上述 `docker-compose` 替换为 `docker compose`。模型无需重转。
两个多实例后端共用新镜像，新路由自动通过现有 Nginx 转发。

文件上传示例（`image` 字段，类别支持重复字段或逗号分隔）：

```bash
curl -fsS http://127.0.0.1:18000/predict-obj-refine/file \
  -F 'image=@test-images/example.jpg' \
  -F 'pre_detect_labels=person' \
  -F 'class_names=helmet' \
  -F 'class_names=vest' \
  -F 'confidence=0.3' \
  -F 'pre_detect_confidence=0.2' \
  -F 'merge_results=true' \
  -F 'return_mask=true' \
  -F 'crop_config_json={"max_crops":4,"max_size":640,"padding":20}'
```

JSON 接口兼容上游的字段名（不同于旧 `/predict` 的 `image`/`class_names`）：

```json
{
  "image_base64": "<图片Base64，支持data URL>",
  "confidence_threshold": 0.3,
  "pre_detect_labels": ["person"],
  "prompts": [{"text": "helmet"}, {"text": "vest"}],
  "return_mask": true,
  "merge_results": true,
  "crop_config": {"max_crops": 4, "max_size": 640, "padding": 20}
}
```

JSON 默认 confidence=0.5、return_mask=false；文件版默认 confidence=0.3、
return_mask=true。`pre_detect_confidence` 可选，未设置时使用精细检测阈值。
标签去除空白、按大小写不敏感去重；精细标签中与预检测标签重复的项被过滤。
空预检测标签回退为 `person`。空精细标签只返回预检测结果，不执行无用裁剪。
仅支持文本，`prompts[].boxes` 非空会返回 422，不会静默忽略几何提示。

兼容上游的合并语义：

| 情况 | 返回结果 |
|---|---|
| `merge_results=true` | 预检测 + 原图精细检测 + 裁剪精细检测，按类别去重 |
| `merge_results=false` | 预检测 + 裁剪精细检测，不执行原图精细检测 |
| 没有预检测目标，`true` | 继续原图精细检测，不裁剪 |
| 没有预检测目标，`false` | 空结果 |

此功能不是目标关联/人员属性判定，不会返回“安全帽属于哪个人”的关系。
ROI 仅限定检测范围；关闭原图精细检测时，ROI 外的精细类别可能被漏掉。

`crop_config` 全部参数：

| 参数 | 默认值 | 作用 |
|---|---|---|
| `max_size` | 640 | 聚合区域尺寸/留白扩张限制；大主体框不被强制截断 |
| `padding` | 20 | 原图像素单位的边缘留白 |
| `w_diou` | 30 | 聚合距离成本权重 |
| `w_expansion` | 5 | 聚合面积扩张成本权重 |
| `count_penalty` | 120 | 增大时更倾向合并、减少裁剪数 |
| `nms_threshold` | 0.2 | 裁剪区域重叠合并阈值，不是检测结果 NMS 阈值 |
| `enable_ar_fix` | true | 尝试修正宽高比，受图像边界和尺寸限制 |
| `target_ar` | 1.0 | 目标宽高比 |
| `max_crops` | 环境变量，默认 4 | 每请求最多处理的裁剪区域 |
| `max_pre_detections` | 环境变量，默认 64 | 最多参与聚合的预检测框 |

在 `.env` 配置容器上限（两个 Compose 文件均已传入环境变量）：

```dotenv
SAM3_REFINE_MAX_CROPS=4
SAM3_REFINE_MAX_PRE_DETECTIONS=64
```

请求可降低上限，但超过容器配置会返回 422。修改 `.env` 后需要
`up -d --no-build --force-recreate`，不需要重新构建。全局允许的最大配置
分别为 32 和 256；生产建议先用少量裁剪。
超过预检测预算时优先用高置信度主体框；超过裁剪预算时优先用包含高置信度
主体的区域。被跳过的区域不执行局部推理，响应 `limited=true` 明确告知降级。
`merge_results=true` 时仍保留原图精细检测，但不能代替被跳过区域的放大检测。

##### OmniCrop 裁剪坐标日志

每次目标细化请求均打印 `[OmniCrop]` 日志。`output_crops` 是引擎生成的候选数，
`selected_crops` 是预算截断后选中的区域数，`limited=1` 表示触及预算。
`pid` 区分不同 Uvicorn worker；`crop[i]` 是浮点裁剪框，`roi[i]` 是真正传入
OpenCV 的整数采样区域。整数原点和宽高分别截断，对齐 TRT 的取整方式，
并做图像边界检查；不足一个像素的无效区域会标记 `valid=0` 并跳过。
这些日志不表示已经成功完成该区域的推理，最终完成数量以 `[Refine] crops=`
及响应 `refinement.crops_processed` 为准。

```text
[OmniCrop] pid=100 input_boxes=8 output_crops=3 selected_crops=3 limited=0 image=1920x1080 max_size=640 max_crops=4
[OmniCrop] pid=100 crop[0]: x1=267.744, y1=255.801, x2=462.256, y2=450.312, w=194.512, h=194.512
[OmniCrop] pid=100 roi[0]: x=267, y=255, w=194, h=194, x2=461, y2=449, valid=1
```

上例是固定 TRT 预检测框的参考值。实际昇腾模型的人员框存在微小数值差异，
裁剪坐标和顺序可能略有不同；对比时按位置匹配，不要只按 `crop[i]` 下标比较。
复现此场景时请求 `max_crops` 至少为 3（默认 4），不要沿用旧示例的 2。

```bash
# 多实例：发起一次目标细化请求后，查看两个后端的日志
docker-compose -f docker-compose.dual.yml logs --tail=200 sam3-npu2 sam3-npu3

# 单实例
docker-compose -f docker-compose.yml logs --tail=200 sam3-service

# 无 NPU 的 C++ 回归测试：固定人员框、TRT 三个裁剪坐标、预算、边界与 NMS
g++ -std=c++17 -O2 -Isrc tests/refine_crop_test.cpp -o /tmp/sam3-refine-crop-test
/tmp/sam3-refine-crop-test
```

响应保留 `results`，额外增加 `refinement` 诊断信息：

```json
{
  "results": [],
  "refinement": {
    "pre_detections": 5,
    "candidate_crops": 3,
    "crops_processed": 2,
    "limited": true,
    "merge_results": true,
    "timings_ms": {
      "pre_detect": 1000,
      "full_refine": 340,
      "crop_plan": 1,
      "crop_refine": 2400,
      "nms": 1,
      "total": 3742
    }
  }
}
```

上述数字仅演示结构，不代表性能承诺。`total` 是 C++ 流程耗时，不含等待实例锁、
图片解码、最终 PNG/RLE 编码和 HTTP 传输；端到端性能以测试脚本的延迟为准。
日志 `[Refine]` 同时打印各阶段耗时、候选/实际裁剪数和是否限额。
框坐标始终为原图像素，mask 仍为框相对的局部掩码，RLE 按行优先排列，
格式为 `[1-based起点,长度,...]`，不是 COCO 的列优先 RLE。

#### 对比普通检测与 obj-refine

测试脚本仍仅依赖 Python 标准库。使用相同图片、类别、并发和 mask 选项对比：

```bash
# 普通检测：原图上检测相同三个类别
python3 scripts/benchmark_service.py --images test-images \
  --class-name person --class-name helmet --class-name vest \
  --concurrency 2 --rounds 3 --warmup 2 \
  --return-mask --save-results --label ordinary --json-output benchmark-ordinary.json

# 精细检测：person 用于定位，helmet/vest 用于局部识别
python3 scripts/benchmark_service.py --images test-images --mode obj-refine \
  --pre-detect-label person --class-name helmet --class-name vest \
  --max-crops 2 --concurrency 2 --rounds 3 --warmup 2 \
  --return-mask --save-results --label refine --json-output benchmark-refine.json
```

`--mode obj-refine` 默认切换到 `/predict-obj-refine/file`；自定义 `--url` 应指向
对应文件接口。增加 `--no-merge-results` 测试仅裁剪路径，
`--pre-detect-confidence 0.2` 可以降低预检测阈值（公平性能对比先保持阈值相同）。
报告增加平均裁剪数、限额请求数和阶段耗时；`--save-results` 会将检测框/RLE
保存到报告供人工准确率检查。不设置该参数则不保留大掩码。
需要有标注的样本才能评价漏检率，不能仅凭“检测数量变多”认定准确率提升。

注意：每个 ROI 仍会缩放到 1008×1008，额外执行一次 Vision 编码。
obj-refine 主要改善局部小目标识别，不保证提高吞吐；裁剪越多通常越慢。
多视频流应按需调用，并结合吞吐、P95、限额比例评估，避免全量帧默认开启。

#### 离线回归测试

无需 NPU 的接口测试（伪模型，用于验证协议和错误码，不代表 OM 推理通过）：

```bash
python3 -m pip install -r service/requirements.txt httpx
python3 -m unittest discover -s tests -v
```

裁剪聚合、限额及 NMS 的独立 C++ 测试，不依赖 CANN/OpenCV：

```bash
g++ -std=c++17 -Wall -Wextra -Isrc tests/refine_crop_test.cpp -o /tmp/sam3-refine-crop-test
/tmp/sam3-refine-crop-test
```

服务器验收应额外覆盖：无主体（两种 merge 分支）、多人/重叠裁剪、边缘主体、
空精细类别、不同预检测阈值、预算触顶、多 worker 并发，以及框/mask 原图对齐。
模型执行失败返回 500；无目标是正常 200 空结果，不能混为一谈。

#### 普通检测

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

