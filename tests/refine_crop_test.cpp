// No CANN/OpenCV needed: g++ -std=c++17 -Isrc tests/refine_crop_test.cpp -o /tmp/refine_crop_test
#include "common/refineCrop.hpp"
#include <cassert>
#include <iostream>
#include <random>

using namespace sam3::refine;

int main()
{
    CropConfig cfg;
    assert(plan_crops({}, 1920, 1080, cfg).crops.empty());
    auto clustered = plan_crops({{100, 100, 160, 280}, {180, 100, 240, 280}}, 1920, 1080, cfg);
    assert(clustered.crops.size() == 1);
    assert(clustered.crops[0].x1 <= 100 && clustered.crops[0].x2 >= 240);
    auto edge = plan_crops({{0, 0, 20, 50}, {1900, 1000, 1920, 1080}}, 1920, 1080, cfg);
    assert(edge.crops.size() == 2);
    for (auto b : edge.crops)
        assert(b.x1 >= 0 && b.y1 >= 0 && b.x2 <= 1920 && b.y2 <= 1080 && b.area() > 0);
    auto large = plan_crops({{0, 0, 1000, 900}}, 1920, 1080, cfg);
    assert(large.crops[0].width() >= 1000 && large.crops[0].height() >= 900);
    auto invalid = plan_crops({{-10, -10, -1, -1}, {5, 5, 2, 2}}, 100, 100, cfg);
    assert(invalid.crops.empty());
    cfg.max_crops = 1;
    auto limited = plan_crops({{10, 10, 30, 80}, {1500, 10, 1520, 80}}, 1920, 1080, cfg);
    assert(limited.limited && limited.candidate_count == 2 && limited.crops.size() == 1);
    cfg.max_pre_detections = 1;
    limited = plan_crops({{10, 10, 30, 80}, {40, 10, 60, 80}}, 1920, 1080, cfg);
    assert(limited.limited && limited.crops.size() == 1);
    cfg.enable_ar_fix = false;
    cfg.padding = 0;
    auto exact = plan_crops({{10, 20, 40, 90}}, 100, 100, cfg);
    assert(exact.crops[0].x1 == 10 && exact.crops[0].y1 == 20 && exact.crops[0].y2 == 90);
    cfg.padding = -1;
    bool threw = false;
    try { plan_crops({}, 100, 100, cfg); } catch (const std::invalid_argument&) { threw = true; }
    assert(threw);

    const std::vector<Rect> boxes{{0, 0, 100, 100}, {1, 1, 101, 101}, {1, 1, 101, 101}, {300, 300, 310, 310}};
    const auto keep = nms_indices(boxes, {0.9f, 0.8f, 0.7f, 0.6f}, {"helmet", "helmet", "person", "helmet"});
    assert((keep == std::vector<size_t>{0, 2, 3}));
    const float nan = std::numeric_limits<float>::quiet_NaN();
    assert(nms_indices({{0, 0, 10, 10}, {nan, 0, 10, 10}}, {nan, 1}, {"x", "x"}).empty());

    // Stress image-boundary/rounding and seed coverage with varied aspect ratios.
    std::mt19937 rng(42);
    for (int i = 0; i < 1000; ++i)
    {
        const int w = 1 + int(rng() % 2000), h = 1 + int(rng() % 1200);
        const int x = int(rng() % w), y = int(rng() % h);
        const Rect seed{float(x), float(y), float(x + 1 + rng() % (w - x)),
                        float(y + 1 + rng() % (h - y))};
        CropConfig varied;
        varied.target_ar = 0.5f + float(rng() % 20) / 10.0f;
        auto plan = plan_crops({seed}, w, h, varied);
        assert(plan.crops.size() == 1);
        const auto b = plan.crops[0];
        assert(b.x1 >= 0 && b.y1 >= 0 && b.x2 <= w && b.y2 <= h);
        assert(b.x1 <= seed.x1 && b.y1 <= seed.y1 && b.x2 >= seed.x2 && b.y2 >= seed.y2);
    }
    std::cout << "refine crop/NMS tests passed\n";
}
