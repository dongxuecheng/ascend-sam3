#pragma once

// Thin Ascend adapter around the pinned, unmodified upstream OmniCrop core.
// Validation and request budgets stay outside the clustering algorithm.
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include "third_party/omnicrop/OmniCrop.hpp"

namespace sam3::refine
{
struct CropConfig
{
    int max_size = 640;
    int padding = 20;
    float w_diou = 30.0f;
    float w_expansion = 5.0f;
    float count_penalty = 120.0f;
    float nms_threshold = 0.2f; // crop overlap merging, NOT detection NMS
    bool enable_ar_fix = true;
    float target_ar = 1.0f;
    int max_crops = 4;
    int max_pre_detections = 64;

    void validate() const
    {
        if (max_size < 1 || max_size > 16384 || padding < 0 || padding > 4096 ||
            max_crops < 1 || max_crops > 32 || max_pre_detections < 1 || max_pre_detections > 256 ||
            !std::isfinite(w_diou) || w_diou < 0 ||
            !std::isfinite(w_expansion) || w_expansion < 0 ||
            !std::isfinite(count_penalty) || count_penalty < 0 ||
            !std::isfinite(nms_threshold) || nms_threshold < 0 || nms_threshold > 1 ||
            !std::isfinite(target_ar) || target_ar < 0.1f || target_ar > 10.0f)
            throw std::invalid_argument("Invalid obj-refine crop configuration");
    }
};

struct Rect
{
    float x1, y1, x2, y2;
    float width() const { return std::max(0.0f, x2 - x1); }
    float height() const { return std::max(0.0f, y2 - y1); }
    float area() const { return width() * height(); }
};

inline float iou(const Rect& a, const Rect& b)
{
    const float inter = std::max(0.0f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1)) *
                        std::max(0.0f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
    const float total = a.area() + b.area() - inter;
    return total > 0 ? inter / total : 0.0f;
}

struct CropPlan
{
    std::vector<Rect> crops;
    int candidate_count = 0;
    bool limited = false;
};

// Preserve caller order. The inference caller ranks seeds by confidence so
// budgets prioritize higher-confidence subjects. Without truncation, crops
// are exactly the upstream engine's floating-point output for these seeds.
inline CropPlan plan_crops(const std::vector<Rect>& ranked_boxes, int width, int height,
                           const CropConfig& cfg)
{
    cfg.validate();
    if (width < 1 || height < 1) throw std::invalid_argument("Invalid image size");
    std::vector<omnicrop::BBox> seeds;
    for (const auto& b : ranked_boxes)
    {
        if (!std::isfinite(b.x1) || !std::isfinite(b.y1) ||
            !std::isfinite(b.x2) || !std::isfinite(b.y2)) continue;
        Rect r{std::clamp(b.x1, 0.0f, float(width)), std::clamp(b.y1, 0.0f, float(height)),
               std::clamp(b.x2, 0.0f, float(width)), std::clamp(b.y2, 0.0f, float(height))};
        if (r.area() > 0) seeds.emplace_back(r.x1, r.y1, r.x2, r.y2);
    }
    CropPlan plan;
    plan.limited = seeds.size() > static_cast<size_t>(cfg.max_pre_detections);
    if (plan.limited) seeds.resize(cfg.max_pre_detections);
    omnicrop::Config upstream;
    upstream.w_diou = cfg.w_diou;
    upstream.w_expansion = cfg.w_expansion;
    upstream.crop_count_penalty = cfg.count_penalty;
    upstream.nms_threshold = cfg.nms_threshold;
    upstream.enable_aspect_ratio_fix = cfg.enable_ar_fix;
    upstream.target_aspect_ratio = cfg.target_ar;
    omnicrop::OmniCropEngine engine(cfg.max_size, cfg.padding);
    for (const auto& crop : engine.cluster_and_crop(seeds, width, height, upstream))
        plan.crops.push_back({crop.x1, crop.y1, crop.x2, crop.y2});
    plan.candidate_count = static_cast<int>(plan.crops.size());
    if (plan.crops.size() > static_cast<size_t>(cfg.max_crops))
    {
        plan.crops.resize(cfg.max_crops);
        plan.limited = true;
    }
    return plan;
}

struct CropRoi
{
    int x = 0, y = 0, width = 0, height = 0;
    bool valid() const { return width > 0 && height > 0; }
};

// Match TRT: truncate origin and extent separately, not floor/ceil endpoints.
// Use this same helper for both logging and the actual OpenCV ROI extraction.
inline CropRoi crop_roi(const Rect& crop, int image_width, int image_height)
{
    if (image_width < 1 || image_height < 1 ||
        !std::isfinite(crop.x1) || !std::isfinite(crop.y1) ||
        !std::isfinite(crop.x2) || !std::isfinite(crop.y2)) return {};
    CropRoi roi;
    roi.x = static_cast<int>(std::clamp(double(crop.x1), 0.0, double(image_width)));
    roi.y = static_cast<int>(std::clamp(double(crop.y1), 0.0, double(image_height)));
    roi.width = std::min(static_cast<int>(std::min(double(crop.width()), double(image_width))), image_width - roi.x);
    roi.height = std::min(static_cast<int>(std::min(double(crop.height()), double(image_height))), image_height - roi.y);
    return roi;
}

// Same-class detection NMS; indices let callers keep masks attached to boxes.
inline std::vector<size_t> nms_indices(const std::vector<Rect>& boxes,
                                       const std::vector<float>& scores,
                                       const std::vector<std::string>& labels,
                                       float threshold = 0.5f)
{
    if (boxes.size() != scores.size() || boxes.size() != labels.size())
        throw std::invalid_argument("NMS input sizes differ");
    std::vector<size_t> order(boxes.size()), keep;
    std::iota(order.begin(), order.end(), 0);
    order.erase(std::remove_if(order.begin(), order.end(), [&](size_t idx) {
        const auto& b = boxes[idx];
        return !std::isfinite(scores[idx]) || !std::isfinite(b.x1) || !std::isfinite(b.y1) ||
               !std::isfinite(b.x2) || !std::isfinite(b.y2) || b.area() <= 0;
    }), order.end());
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return scores[a] > scores[b]; });
    for (size_t idx : order)
    {
        bool suppressed = false;
        for (size_t prev : keep)
            if (labels[idx] == labels[prev] && iou(boxes[idx], boxes[prev]) > threshold)
            { suppressed = true; break; }
        if (!suppressed) keep.push_back(idx);
    }
    return keep;
}
} // namespace sam3::refine
