// No CANN/OpenCV needed. Assertions stay enabled in release builds.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "common/trtSampling.hpp"
#include <cassert>
#include <iostream>
#include <random>
#include <vector>

using namespace sam3::trt;

static float reference_pixel(const uint8_t* image, int stride, int w, int h,
                             float x, float y, int channel)
{
    if (x <= -1 || x >= w || y <= -1 || y >= h) return 114;
    const int left = static_cast<int>(std::floor(x)), top = static_cast<int>(std::floor(y));
    const float lx = x - left, ly = y - top;
    auto at = [&](int xx, int yy) -> float {
        return xx < 0 || xx >= w || yy < 0 || yy >= h ? 114 : image[yy * stride + xx * 3 + channel];
    };
    const float a = (1 - ly) * (1 - lx) * at(left, top);
    const float b = (1 - ly) * lx * at(left + 1, top);
    const float c = ly * (1 - lx) * at(left, top + 1);
    const float d = ly * lx * at(left + 1, top + 1);
    return std::floor(a + b + c + d + 0.5f);
}

static uint8_t reference_mask(const std::vector<float>& logits, int w, int h, float x, float y)
{
    if (x < 0 || x >= w || y < 0 || y >= h) return 0;
    const int ix = static_cast<int>(x), iy = static_cast<int>(y);
    const float fx = x - ix, fy = y - iy;
    auto at = [&](int xx, int yy) { return xx < w && yy < h ? logits[yy * w + xx] : 0.0f; };
    const float value = (1 - fy) * (1 - fx) * at(ix, iy) + (1 - fy) * fx * at(ix + 1, iy) +
                         fy * (1 - fx) * at(ix, iy + 1) + fy * fx * at(ix + 1, iy + 1);
    return value > 0.5f ? 255 : 0;
}

static void matrices()
{
    const auto full = resize_inverse(4, 2, 8, 4);
    assert(full[0] == 0.5f && full[4] == 0.5f && full[2] == -0.25f && full[5] == -0.25f);
    const auto crop = crop_inverse(1, 0, 2, 1, 4, 1);
    assert(crop[0] == 0.5f && crop[2] == 1 && crop[4] == 1 && crop[5] == 0);
    const auto mask = mask_inverse(4, 2, 8, 4, 3, 1);
    assert(mask[0] == 0.5f && mask[2] == 1.25f && mask[5] == 0.25f);
    for (const auto& roi : std::vector<std::array<int, 4>>{{267, 255, 194, 194},
                                                       {952, 297, 314, 314},
                                                       {1234, 295, 383, 383}})
    {
        const auto matrix = crop_inverse(roi[0], roi[1], roi[2], roi[3], 1008, 1008);
        assert(std::abs(matrix[0] - static_cast<float>(roi[2]) / 1008) < 1e-6f);
        assert(std::abs(matrix[2] - roi[0]) < 0.0002f);
        assert(std::abs(matrix[5] - roi[1]) < 0.0002f);
    }
    bool rejected = false;
    try { crop_inverse(0, 0, 0, 1, 1, 1); } catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
    rejected = false;
    try { resize_inverse(1, 1, 0, 1); } catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
}

static void crop_uses_original_neighbors()
{
    // Crop contains x=1,2 only. Its last sample must use original x=3 too.
    const uint8_t source[] = {0, 1, 2, 40, 41, 42, 120, 121, 122, 240, 241, 242};
    uint8_t out[12] = {};
    const auto matrix = crop_inverse(1, 0, 2, 1, 4, 1);
    warp_bgr_rows(source, 12, 4, 1, out, 12, 4, matrix, 0, 1);
    const uint8_t expected[] = {40, 41, 42, 80, 81, 82, 120, 121, 122, 180, 181, 182};
    for (int i = 0; i < 12; ++i) assert(out[i] == expected[i]);

    // Full-image upsampling blends border=114, not OpenCV's replicated edge.
    const std::vector<uint8_t> constant(2 * 2 * 3, 200);
    std::vector<uint8_t> enlarged(4 * 4 * 3);
    warp_bgr_rows(constant.data(), 6, 2, 2, enlarged.data(), 12, 4, resize_inverse(2, 2, 4, 4), 0, 4);
    assert(enlarged.front() == 162 && enlarged.back() == 162);
}

static void raw_mask_threshold_and_components()
{
    // Equality at 0.5 is background. A disconnected island and a hole remain.
    std::vector<float> mask{0.6f, 0.5f, 0.6f, 0.6f, 0.1f, 0.6f, 0.6f, 0.6f, 0.6f};
    uint8_t out[9] = {};
    warp_mask_rows(mask.data(), 3, 3, out, 3, 3, resize_inverse(3, 3, 3, 3), 0, 3);
    assert(out[0] == 255 && out[1] == 0 && out[2] == 255 && out[4] == 0);
    std::vector<float> islands{1, 0, 0, 0, 0, 1};
    uint8_t separated[6] = {};
    warp_mask_rows(islands.data(), 6, 1, separated, 6, 6, resize_inverse(6, 1, 6, 1), 0, 1);
    assert(separated[0] == 255 && separated[5] == 255);
    std::vector<float> positive(4, 0.25f);
    uint8_t not_sigmoid[4] = {};
    warp_mask_rows(positive.data(), 2, 2, not_sigmoid, 2, 2, resize_inverse(2, 2, 2, 2), 0, 2);
    for (auto pixel : not_sigmoid) assert(pixel == 0);
}

static void mask_global_coordinates()
{
    const std::vector<float> logits{1, 0, 0, 1};
    const auto base = resize_inverse(2, 2, 4, 4);
    uint8_t full[16] = {}, box[4] = {};
    warp_mask_rows(logits.data(), 2, 2, full, 4, 4, base, 0, 4);
    warp_mask_rows(logits.data(), 2, 2, box, 2, 2, mask_inverse(2, 2, 4, 4, 1, 1), 0, 2);
    assert(full[0] == 0); // negative half-pixel coordinates rejected outright
    for (int y = 0; y < 2; ++y)
        for (int x = 0; x < 2; ++x) assert(box[y * 2 + x] == full[(y + 1) * 4 + x + 1]);
    const Affine outside{1, 0, -0.001f, 0, 1, 0};
    uint8_t sample = 255;
    warp_mask_rows(logits.data(), 2, 2, &sample, 1, 1, outside, 0, 1);
    assert(sample == 0);
    const Affine right_edge{1, 0, 1.5f, 0, 1, 1};
    warp_mask_rows(logits.data(), 2, 2, &sample, 1, 1, right_edge, 0, 1);
    assert(sample == 0); // half of logit=1 is exactly 0.5, not foreground
}

static void random_reference_comparison()
{
    std::mt19937 random(514);
    for (int iteration = 0; iteration < 500; ++iteration)
    {
        const int w = 1 + random() % 35, h = 1 + random() % 27;
        const int dw = 1 + random() % 43, dh = 1 + random() % 37;
        const int stride = w * 3 + 7, dest_stride = dw * 3 + 5;
        std::vector<uint8_t> source(stride * h), dest(dest_stride * dh, 99);
        for (auto& v : source) v = random() % 256;
        const auto matrix = iteration % 2 ? resize_inverse(w, h, dw, dh) :
            crop_inverse(w / 3, h / 3, std::max(1, w / 2), std::max(1, h / 2), dw, dh);
        // Test the row-sliced primitive actually used by cv::parallel_for_.
        warp_bgr_rows(source.data(), stride, w, h, dest.data(), dest_stride, dw, matrix, 0, dh / 2);
        warp_bgr_rows(source.data(), stride, w, h, dest.data(), dest_stride, dw, matrix, dh / 2, dh);
        for (int y = 0; y < dh; ++y)
        {
            for (int x = 0; x < dw; ++x)
                for (int c = 0; c < 3; ++c)
                    assert(dest[y * dest_stride + x * 3 + c] == reference_pixel(
                        source.data(), stride, w, h, matrix[0] * x + matrix[1] * y + matrix[2],
                        matrix[3] * x + matrix[4] * y + matrix[5], c));
            for (int x = dw * 3; x < dest_stride; ++x) assert(dest[y * dest_stride + x] == 99);
        }

        std::vector<float> logits(w * h);
        for (auto& v : logits) v = (static_cast<int>(random() % 2001) - 1000) / 300.0f;
        std::vector<uint8_t> mask((dw + 3) * dh, 99);
        const auto mask_matrix = mask_inverse(w, h, dw * 2, dh * 2, dw / 2, dh / 2);
        warp_mask_rows(logits.data(), w, h, mask.data(), dw + 3, dw, mask_matrix, 0, dh);
        for (int y = 0; y < dh; ++y)
        {
            for (int x = 0; x < dw; ++x)
                assert(mask[y * (dw + 3) + x] == reference_mask(logits, w, h,
                    mask_matrix[0] * x + mask_matrix[1] * y + mask_matrix[2],
                    mask_matrix[3] * x + mask_matrix[4] * y + mask_matrix[5]));
            for (int x = dw; x < dw + 3; ++x) assert(mask[y * (dw + 3) + x] == 99);
        }
    }
}

int main()
{
    matrices();
    crop_uses_original_neighbors();
    raw_mask_threshold_and_components();
    mask_global_coordinates();
    random_reference_comparison();
    std::cout << "TRT sampling tests passed (500 randomized image/mask cases)\n";
}
