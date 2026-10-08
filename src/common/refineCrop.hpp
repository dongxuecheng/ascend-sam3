#pragma once

// Independently implemented CPU crop clustering; no TensorRT/CUDA dependency.
// The knobs follow the upstream obj-refine API, but crops are not guaranteed
// to be identical to upstream OmniCrop's iterative optimizer.
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

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

inline Rect unite(const Rect& a, const Rect& b)
{
    return {std::min(a.x1, b.x1), std::min(a.y1, b.y1),
            std::max(a.x2, b.x2), std::max(a.y2, b.y2)};
}

inline float iou(const Rect& a, const Rect& b)
{
    const float inter = std::max(0.0f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1)) *
                        std::max(0.0f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
    const float total = a.area() + b.area() - inter;
    return total > 0 ? inter / total : 0.0f;
}

inline Rect finalize(const Rect& box, int width, int height, const CropConfig& cfg)
{
    float w = box.width() + 2.0f * cfg.padding;
    float h = box.height() + 2.0f * cfg.padding;
    if (cfg.enable_ar_fix)
    {
        w = std::max(w, h * cfg.target_ar);
        h = std::max(h, w / cfg.target_ar);
    }
    // max_size limits expansion/merging, never cuts off a large seed object.
    w = std::min(static_cast<float>(width), std::max(box.width(), std::min(w, float(cfg.max_size))));
    h = std::min(static_cast<float>(height), std::max(box.height(), std::min(h, float(cfg.max_size))));
    const float x = std::clamp((box.x1 + box.x2 - w) * 0.5f, 0.0f, float(width) - w);
    const float y = std::clamp((box.y1 + box.y2 - h) * 0.5f, 0.0f, float(height) - h);
    // Cover every seed pixel; these integer bounds are also used for offsets.
    return {std::floor(x), std::floor(y), std::min(float(width), std::ceil(x + w)),
            std::min(float(height), std::ceil(y + h))};
}

struct CropPlan
{
    std::vector<Rect> crops;
    int candidate_count = 0;
    bool limited = false;
};

// Input ordered by confidence: under the budget, lower-confidence clusters
// are skipped explicitly (reported by CropPlan::limited), never silently.
inline CropPlan plan_crops(const std::vector<Rect>& ranked_boxes, int width, int height,
                           const CropConfig& cfg)
{
    cfg.validate();
    if (width < 1 || height < 1) throw std::invalid_argument("Invalid image size");
    std::vector<Rect> seeds;
    for (const auto& b : ranked_boxes)
    {
        if (!std::isfinite(b.x1) || !std::isfinite(b.y1) ||
            !std::isfinite(b.x2) || !std::isfinite(b.y2)) continue;
        Rect r{std::clamp(b.x1, 0.0f, float(width)), std::clamp(b.y1, 0.0f, float(height)),
               std::clamp(b.x2, 0.0f, float(width)), std::clamp(b.y2, 0.0f, float(height))};
        if (r.area() > 0) seeds.push_back(r);
    }
    CropPlan plan;
    plan.limited = seeds.size() > static_cast<size_t>(cfg.max_pre_detections);
    if (plan.limited) seeds.resize(cfg.max_pre_detections);
    // Greedy agglomeration balances normalized expansion/distance costs
    // against a per-crop penalty. Earliest/highest-confidence seed wins order.
    while (seeds.size() > 1)
    {
        size_t best_i = 0, best_j = 0;
        float best = std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < seeds.size(); ++i)
            for (size_t j = i + 1; j < seeds.size(); ++j)
            {
                const Rect joined = unite(seeds[i], seeds[j]);
                if (joined.width() > cfg.max_size || joined.height() > cfg.max_size) continue;
                const float dx = (seeds[i].x1 + seeds[i].x2 - seeds[j].x1 - seeds[j].x2) * 0.5f;
                const float dy = (seeds[i].y1 + seeds[i].y2 - seeds[j].y1 - seeds[j].y2) * 0.5f;
                const float diagonal = joined.width() * joined.width() + joined.height() * joined.height();
                const float expansion = joined.area() / std::max(1.0f, seeds[i].area() + seeds[j].area());
                const float cost = cfg.w_diou * (dx * dx + dy * dy) / std::max(1.0f, diagonal) +
                                   cfg.w_expansion * std::max(0.0f, expansion - 1.0f) - cfg.count_penalty;
                if (cost < best && cost <= 0) { best = cost; best_i = i; best_j = j; }
            }
        if (best_j == best_i) break;
        seeds[best_i] = unite(seeds[best_i], seeds[best_j]);
        seeds.erase(seeds.begin() + best_j);
    }
    for (const auto& seed : seeds) plan.crops.push_back(finalize(seed, width, height, cfg));
    // Merge overlapping expanded crops when the union remains within budget.
    bool changed = true;
    while (changed)
    {
        changed = false;
        for (size_t i = 0; i < plan.crops.size() && !changed; ++i)
            for (size_t j = i + 1; j < plan.crops.size(); ++j)
            {
                const Rect joined = unite(plan.crops[i], plan.crops[j]);
                if (iou(plan.crops[i], plan.crops[j]) > cfg.nms_threshold &&
                    joined.width() <= cfg.max_size && joined.height() <= cfg.max_size)
                {
                    plan.crops[i] = joined;
                    plan.crops.erase(plan.crops.begin() + j);
                    changed = true;
                    break;
                }
            }
    }
    plan.candidate_count = static_cast<int>(plan.crops.size());
    if (plan.crops.size() > static_cast<size_t>(cfg.max_crops))
    {
        plan.crops.resize(cfg.max_crops);
        plan.limited = true;
    }
    return plan;
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
