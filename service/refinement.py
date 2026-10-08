"""Obj-refine protocol validation, independent of the native NPU extension."""

import os
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _limit(name: str, default: int, maximum: int) -> int:
    value = int(os.getenv(name, str(default)))
    if not 1 <= value <= maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}")
    return value


MAX_CROPS = _limit("SAM3_REFINE_MAX_CROPS", 4, 32)
MAX_PRE_DETECTIONS = _limit("SAM3_REFINE_MAX_PRE_DETECTIONS", 64, 256)


class CropConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    max_size: int = Field(default=640, ge=1, le=16384)
    padding: int = Field(default=20, ge=0, le=4096)
    w_diou: float = Field(default=30.0, ge=0, le=100000)
    w_expansion: float = Field(default=5.0, ge=0, le=100000)
    count_penalty: float = Field(default=120.0, ge=0, le=100000)
    nms_threshold: float = Field(default=0.2, ge=0, le=1)
    enable_ar_fix: bool = True
    target_ar: float = Field(default=1.0, ge=0.1, le=10)
    max_crops: int = Field(default=MAX_CROPS, ge=1, le=MAX_CROPS)
    max_pre_detections: int = Field(default=MAX_PRE_DETECTIONS, ge=1, le=MAX_PRE_DETECTIONS)


def normalize_labels(values: List[str]) -> List[str]:
    labels, seen = [], set()
    for value in values:
        label = value.strip()
        if not label:
            continue
        if len(label) > 256:
            raise ValueError("Each text label must be at most 256 characters")
        if label.lower() not in seen:
            labels.append(label)
            seen.add(label.lower())
    if len(labels) > 32:
        raise ValueError("At most 32 unique labels per stage are supported")
    return labels


class RefinePrompt(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = ""
    # Accept upstream's empty boxes, reject real geometric prompts explicitly.
    boxes: Optional[List[dict]] = Field(default_factory=list)

    @model_validator(mode="after")
    def text_only(self):
        if self.boxes:
            raise ValueError("obj-refine only supports text prompts; boxes are not supported")
        return self


class RefineRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    image_base64: str = Field(min_length=1)
    confidence_threshold: float = Field(default=0.5, ge=0, le=1)
    pre_detect_confidence: Optional[float] = Field(default=None, ge=0, le=1)
    pre_detect_labels: List[str] = Field(default_factory=lambda: ["person"], max_length=32)
    prompts: List[RefinePrompt] = Field(max_length=32)
    return_mask: bool = False
    merge_results: bool = True
    crop_config: Optional[CropConfig] = None

    @model_validator(mode="after")
    def clean_labels(self):
        # Upstream falls back to person for an empty pre-detection list.
        self.pre_detect_labels = normalize_labels(self.pre_detect_labels) or ["person"]
        normalize_labels([p.text for p in self.prompts])
        return self

    def refine_labels(self) -> List[str]:
        excluded = {label.lower() for label in self.pre_detect_labels}
        return [label for label in normalize_labels([p.text for p in self.prompts])
                if label.lower() not in excluded]
