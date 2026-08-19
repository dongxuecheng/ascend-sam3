"""
SAM3 FastAPI 推理服务（基于 pybind11 封装的 ascendsam3）

提供两个接口：
- POST /detect/file    上传图片文件
- POST /predict        传入 base64 编码图片

请求参数：
- class_names: 检测类别文本列表，例如 ["person", "car"]
- confidence: 置信度阈值，默认 0.3
- return_mask: 是否返回 mask（PNG 字节，base64 编码），默认 true

返回：
- num_detections: 检测数量
- elapsed_ms: 推理耗时（毫秒）
- boxes: 检测框列表，每项包含 class_name、score、box、mask_png（可选）
"""

import base64
import os
import time
from typing import List, Optional

import cv2
import numpy as np

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# 尝试导入 pybind11 模块；如果 build/ 下没有，则尝试 PYTHONPATH 中已安装的版本
import sys

BUILD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build")
if BUILD_DIR not in sys.path:
    sys.path.insert(0, BUILD_DIR)

import ascendsam3

app = FastAPI(title="SAM3 Ascend Inference Service", version="1.1.0")


@app.middleware("http")
async def add_worker_identity(request: Request, call_next):
    """Expose the serving process so multi-worker balancing can be verified."""
    response = await call_next(request)
    response.headers["X-SAM3-Worker-PID"] = str(os.getpid())
    return response

# 挂载静态文件目录，前端页面放在 service/static/
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def root():
    """返回前端页面"""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

# 模型与资源路径，可通过环境变量覆盖
VISION_MODEL = os.getenv("VISION_MODEL", "models/om-models/vision-encoder.om")
TEXT_MODEL = os.getenv("TEXT_MODEL", "models/om-models/text-encoder.om")
DECODER_MODEL = os.getenv("DECODER_MODEL", "models/om-models/decoder_static.om")
FPN_POS2 = os.getenv("FPN_POS2", "models/om-models/fpn_pos_2_constant.npy")
TOKENIZER = os.getenv("TOKENIZER", "models/onnx-models/tokenizer.json")


class DetectBase64Request(BaseModel):
    image: str = Field(..., description="base64 编码的图片，支持 data URL")
    class_names: List[str] = Field(default=["person"], description="检测类别列表")
    confidence: float = Field(default=0.3, ge=0.0, le=1.0)
    return_mask: bool = Field(default=True, description="是否返回 mask PNG")


# 全局单例模型，服务启动时加载一次
_model: Optional[ascendsam3.Sam3Model] = None


def _load_model() -> ascendsam3.Sam3Model:
    global _model
    if _model is None:
        for p in [VISION_MODEL, TEXT_MODEL, DECODER_MODEL, FPN_POS2, TOKENIZER]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing required file: {p}")
        _model = ascendsam3.Sam3Model(
            VISION_MODEL,
            TEXT_MODEL,
            DECODER_MODEL,
            FPN_POS2,
            TOKENIZER,
        )
    return _model


def _png_to_rle(png_bytes: bytes) -> tuple:
    """将 PNG 字节解码为 RLE，返回 (RLE, width, height)

    注意：C++ 后端返回的 mask 已经是基于当前检测框的，因此这里直接编码全图，
    前端绘制时放在 box 左上角即可。
    """
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    mask = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError("Failed to decode PNG mask")
    binary = (mask > 0).astype(np.uint8)
    flat = np.concatenate([[0], binary.ravel(), [0]])
    runs = np.where(flat[1:] != flat[:-1])[0] + 1
    runs[1::2] -= runs[0::2]
    h, w = binary.shape
    return runs.tolist(), w, h


def _decode_base64_image(b64: str) -> bytes:
    """解码 base64 图片，支持 data URL"""
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    return base64.b64decode(b64)


def _detect(image_bytes: bytes, class_names: List[str], confidence: float, return_mask: bool) -> dict:
    model = _load_model()
    raw_results = model.detect(image_bytes, class_names, confidence, return_mask)

    results = []
    for r in raw_results:
        raw_box = r.get("box")
        
        # 将字典 {"left": x, "top": y, "right": z, "bottom": w} 转换为列表 [left, top, right, bottom]
        if isinstance(raw_box, dict):
            box_list = [
                float(raw_box.get("left", 0.0)),
                float(raw_box.get("top", 0.0)),
                float(raw_box.get("right", 0.0)),
                float(raw_box.get("bottom", 0.0))
            ]
        elif isinstance(raw_box, (list, tuple)):
            box_list = [float(x) for x in raw_box]
        else:
            box_list = []

        item = {
            "label": r["class_name"],  # 将 "class_name" 映射为 "label"
            "score": r["score"],
            "box": box_list,           # 使用适配好的列表格式
        }

        if return_mask and "mask_png" in r:
            try:
                # C++ 后端返回的 mask 已是 box 相对，直接转 RLE
                rle, mw, mh = _png_to_rle(r["mask_png"])
                item["mask"] = rle
                item["mask_width"] = mw
                item["mask_height"] = mh
            except Exception:
                # 容错处理：当编解码失败时，设为 None
                item["mask"] = None
                item["mask_width"] = 0
                item["mask_height"] = 0
        results.append(item)

    return {
        "results": results
    }


@app.on_event("startup")
def startup():
    # 服务启动时预热加载模型
    _load_model()
    print(
        f"SAM3 worker ready: pid={os.getpid()}, "
        f"configured_workers={os.getenv('SAM3_WORKERS', '1')}",
        flush=True,
    )


@app.get("/health")
def health():
    try:
        _load_model()
        return {
            "status": "ok",
            "device_id": int(os.getenv("ASCEND_DEVICE_ID", "0")),
            "worker_pid": os.getpid(),
            "configured_workers": int(os.getenv("SAM3_WORKERS", "1")),
        }
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "unhealthy", "error": str(e)})


@app.post("/predict/file")
def detect_file(
    image: UploadFile = File(...),
    # 将类型改为 List[str]，默认值设为包含 "person" 的列表
    class_names: List[str] = Form(["person"]), 
    confidence: float = Form(0.3),
    return_mask: bool = Form(True),
):
    """
    上传图片文件进行检测。
    class_names 接收一个列表，客户端可以通过传递多个同名表单字段来传参。
    """
    try:
        image_bytes = image.file.read()
        # 同时支持多个 class_names 表单字段，以及单个逗号分隔字段
        if len(class_names) == 1 and ',' in class_names[0]:
            names = [n.strip() for n in class_names[0].split(',') if n.strip()]
        else:
            names = [n.strip() for n in class_names if n.strip()]
        
        print(f"[DEBUG] received class_names={class_names}, parsed names={names}")
        result = _detect(image_bytes, names, confidence, return_mask)
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/predict")
def detect_base64(req: DetectBase64Request):
    """传入 base64 编码图片进行检测"""
    try:
        image_bytes = _decode_base64_image(req.image)
        result = _detect(image_bytes, req.class_names, req.confidence, req.return_mask)
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
