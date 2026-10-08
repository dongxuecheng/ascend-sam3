// No CANN/OpenCV needed: g++ -std=c++17 -Isrc tests/refine_crop_test.cpp -o /tmp/refine_crop_test
// Test assertions must remain enabled even with release-oriented compilers.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "common/refineCrop.hpp"
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <random>

using namespace sam3::refine;

static void assert_close(float actual, float expected, float tolerance = 0.002f)
{
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance)
    {
        std::cerr << "coordinate mismatch: " << actual << " expected " << expected << '\n';
        std::abort();
    }
}

static void test_trt_crop_regression()
{
    // Eight TRT pre-detections supplied by the user, in returned order.
    // Image bounds do not clip the three reference crops in this fixture.
    const std::vector<Rect> seeds{
        {335.625f, 275.80078125f, 394.375f, 430.3125f},
        {1254.375f, 329.4140625f, 1333.75f, 590.2734375f},
        {1510.f, 352.6171875f, 1597.5f, 638.0859375f},
        {986.875f, 317.8125f, 1099.375f, 592.734375f},
        {1144.375f, 332.578125f, 1233.125f, 591.328125f},
        {1386.25f, 357.01171875f, 1458.75f, 618.046875f},
        {1321.25f, 344.53125f, 1416.25f, 649.6875f},
        {1433.75f, 324.84375f, 1508.75f, 574.453125f},
    };
    const std::vector<Rect> expected{
        {267.744140625f, 255.80078125f, 462.255859375f, 450.3125f},
        {952.5390625f, 297.8125f, 1267.4609375f, 612.734375f},
        {1234.375f, 295.703125f, 1617.5f, 678.828125f},
    };
    auto plan = plan_crops(seeds, 1920, 1080, CropConfig{});
    assert(plan.candidate_count == 3 && plan.crops.size() == 3 && !plan.limited);
    std::sort(plan.crops.begin(), plan.crops.end(), [](const Rect& a, const Rect& b) { return a.x1 < b.x1; });
    const std::vector<CropRoi> expected_rois{{267, 255, 194, 194}, {952, 297, 314, 314}, {1234, 295, 383, 383}};
    for (size_t i = 0; i < expected.size(); ++i)
    {
        const auto& actual = plan.crops[i];
        assert_close(actual.x1, expected[i].x1); assert_close(actual.y1, expected[i].y1);
        assert_close(actual.x2, expected[i].x2); assert_close(actual.y2, expected[i].y2);
        const auto roi = crop_roi(actual, 1920, 1080);
        assert(roi.x == expected_rois[i].x && roi.y == expected_rois[i].y);
        assert(roi.width == expected_rois[i].width && roi.height == expected_rois[i].height);
    }
    CropConfig limited_cfg;
    limited_cfg.max_crops = 1;
    const auto limited = plan_crops(seeds, 1920, 1080, limited_cfg);
    assert(limited.limited && limited.candidate_count == 3 && limited.crops.size() == 1);

    // Tiny Ascend pre-detection differences should still produce three crops.
    auto ascend_seeds = seeds;
    ascend_seeds[3] = {987.5f, 317.4609375f, 1098.75f, 592.3828125f};
    ascend_seeds[4].y1 = 332.40234375f;
    ascend_seeds[5] = {1386.25f, 356.66015625f, 1458.75f, 618.75f};
    assert(plan_crops(ascend_seeds, 1920, 1080, CropConfig{}).crops.size() == 3);
    // Production ranks seeds by confidence, unlike TRT's decoder query order.
    std::vector<Rect> ranked_ascend;
    for (size_t index : {3u, 4u, 6u, 1u, 2u, 5u, 0u, 7u})
        ranked_ascend.push_back(ascend_seeds[index]);
    assert(plan_crops(ranked_ascend, 1920, 1080, CropConfig{}).crops.size() == 3);
}

static void test_parameter_mapping()
{
    const std::vector<Rect> seeds{{80, 50, 110, 150}, {170, 90, 210, 200}, {560, 300, 600, 450}};
    for (bool aspect_fix : {false, true})
    {
        CropConfig cfg;
        cfg.max_size = 500; cfg.padding = 17; cfg.w_diou = 7.5f;
        cfg.w_expansion = 11; cfg.count_penalty = 25; cfg.nms_threshold = 0.6f;
        cfg.enable_ar_fix = aspect_fix; cfg.target_ar = 1.7f;
        omnicrop::Config raw;
        raw.w_diou = cfg.w_diou; raw.w_expansion = cfg.w_expansion;
        raw.crop_count_penalty = cfg.count_penalty; raw.nms_threshold = cfg.nms_threshold;
        raw.enable_aspect_ratio_fix = cfg.enable_ar_fix; raw.target_aspect_ratio = cfg.target_ar;
        std::vector<omnicrop::BBox> raw_seeds;
        for (const auto& b : seeds) raw_seeds.emplace_back(b.x1, b.y1, b.x2, b.y2);
        const auto expected = omnicrop::OmniCropEngine(cfg.max_size, cfg.padding).cluster_and_crop(raw_seeds, 800, 600, raw);
        const auto actual = plan_crops(seeds, 800, 600, cfg);
        assert(actual.crops.size() == expected.size());
        for (size_t i = 0; i < expected.size(); ++i)
        {
            assert(actual.crops[i].x1 == expected[i].x1 && actual.crops[i].x2 == expected[i].x2);
            assert(actual.crops[i].y1 == expected[i].y1 && actual.crops[i].y2 == expected[i].y2);
        }
    }
}

int main()
{
    test_trt_crop_regression();
    test_parameter_mapping();
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
    const float invalid_coord = std::numeric_limits<float>::infinity();
    assert(plan_crops({{invalid_coord, 0, 10, 10}}, 100, 100, cfg).crops.empty());
    assert(!crop_roi({invalid_coord, 0, 10, 10}, 100, 100).valid());
    assert(!crop_roi({5, 5, 5.5f, 6}, 100, 100).valid());
    const auto clipped_roi = crop_roi({95.5f, 97.5f, 105.5f, 110.5f}, 100, 100);
    assert(clipped_roi.x == 95 && clipped_roi.y == 97 && clipped_roi.width == 5 && clipped_roi.height == 3);
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
        constexpr float eps = 0.001f;
        assert(b.x1 >= -eps && b.y1 >= -eps && b.x2 <= w + eps && b.y2 <= h + eps);
        assert(b.x1 <= seed.x1 + eps && b.y1 <= seed.y1 + eps && b.x2 >= seed.x2 - eps && b.y2 >= seed.y2 - eps);
        const auto roi = crop_roi(b, w, h);
        if (roi.valid()) assert(roi.x >= 0 && roi.y >= 0 && roi.x + roi.width <= w && roi.y + roi.height <= h);
    }
    std::cout << "OmniCrop TRT-coordinate, parameter mapping, budget, ROI, boundary and NMS tests passed\n";
}
