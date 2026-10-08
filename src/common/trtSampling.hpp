#pragma once

// CPU equivalents of the sampling semantics in trt-sam3 at commit
// 593b4ae199c23e6a2042a7dff268c79a9904b253:
// src/common/affine.hpp and src/kernels/preprocess.cu.
// Keep this independent of ACL/OpenCV so the exact runtime sampler is testable.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace sam3::trt
{
using Affine = std::array<float, 6>; // destination -> source

inline Affine resize_inverse(int source_w, int source_h, int dest_w, int dest_h)
{
    if (source_w <= 0 || source_h <= 0 || dest_w <= 0 || dest_h <= 0)
        throw std::invalid_argument("Invalid resize dimensions");
    // Match TRT's float reciprocal, including half-pixel center alignment.
    const float sx = 1.0f / (static_cast<float>(dest_w) / source_w);
    const float sy = 1.0f / (static_cast<float>(dest_h) / source_h);
    return {sx, 0, 0.5f * sx - 0.5f, 0, sy, 0.5f * sy - 0.5f};
}

inline Affine crop_inverse(int x, int y, int crop_w, int crop_h, int dest_w, int dest_h)
{
    if (crop_w <= 0 || crop_h <= 0 || dest_w <= 0 || dest_h <= 0)
        throw std::invalid_argument("Invalid crop dimensions");
    // CropResizeMatrix has NO half-pixel compensation. Invert its float i2d
    // in double as upstream does (rather than substituting an ROI cv::resize).
    const float sx = static_cast<float>(dest_w) / crop_w;
    const float sy = static_cast<float>(dest_h) / crop_h;
    const float tx = -sx * x, ty = -sy * y;
    const double inverse_det = 1.0 / static_cast<double>(sx * sy);
    const double a = sy * inverse_det, b = sx * inverse_det;
    return {static_cast<float>(a), 0, static_cast<float>(-a * tx),
            0, static_cast<float>(b), static_cast<float>(-b * ty)};
}

inline Affine mask_inverse(int mask_w, int mask_h, int image_w, int image_h, int box_x, int box_y)
{
    auto matrix = resize_inverse(mask_w, mask_h, image_w, image_h);
    matrix[2] = matrix[0] * box_x + matrix[1] * box_y + matrix[2];
    matrix[5] = matrix[3] * box_x + matrix[4] * box_y + matrix[5];
    return matrix;
}

inline void warp_bgr_rows(const uint8_t* src, std::size_t src_stride, int src_w, int src_h,
                          uint8_t* dst, std::size_t dst_stride, int dest_w,
                          const Affine& matrix, int row_begin, int row_end, uint8_t border = 114)
{
    for (int dy = row_begin; dy < row_end; ++dy)
    {
        auto* out = dst + static_cast<std::size_t>(dy) * dst_stride;
        for (int dx = 0; dx < dest_w; ++dx)
        {
            const float x = matrix[0] * dx + matrix[1] * dy + matrix[2];
            const float y = matrix[3] * dx + matrix[4] * dy + matrix[5];
            if (x <= -1 || x >= src_w || y <= -1 || y >= src_h)
            {
                out[dx * 3] = out[dx * 3 + 1] = out[dx * 3 + 2] = border;
                continue;
            }
            const int x0 = static_cast<int>(std::floor(x)), y0 = static_cast<int>(std::floor(y));
            const float lx = x - x0, ly = y - y0, hx = 1 - lx, hy = 1 - ly;
            const float weights[4] = {hy * hx, hy * lx, ly * hx, ly * lx};
            const uint8_t fill[3] = {border, border, border};
            const uint8_t* taps[4] = {fill, fill, fill, fill};
            for (int j = 0; j < 4; ++j)
            {
                const int xx = x0 + (j & 1), yy = y0 + (j >> 1);
                if (xx >= 0 && xx < src_w && yy >= 0 && yy < src_h)
                    taps[j] = src + static_cast<std::size_t>(yy) * src_stride + xx * 3;
            }
            for (int c = 0; c < 3; ++c)
            {
                const float value = weights[0] * taps[0][c] + weights[1] * taps[1][c] +
                                    weights[2] * taps[2][c] + weights[3] * taps[3][c];
                out[dx * 3 + c] = static_cast<uint8_t>(std::floor(value + 0.5f));
            }
        }
    }
}

inline void warp_mask_rows(const float* src, int src_w, int src_h,
                           uint8_t* dst, std::size_t dst_stride, int dest_w,
                           const Affine& matrix, int row_begin, int row_end)
{
    for (int dy = row_begin; dy < row_end; ++dy)
    {
        auto* out = dst + static_cast<std::size_t>(dy) * dst_stride;
        for (int dx = 0; dx < dest_w; ++dx)
        {
            const float x = matrix[0] * dx + matrix[1] * dy + matrix[2];
            const float y = matrix[3] * dx + matrix[4] * dy + matrix[5];
            // Unlike the BGR sampler, TRT's mask sampler rejects negative
            // coordinates outright. The border value is zero.
            if (x < 0 || x >= src_w || y < 0 || y >= src_h)
            {
                out[dx] = 0;
                continue;
            }
            const int x0 = static_cast<int>(x), y0 = static_cast<int>(y);
            const float lx = x - x0, ly = y - y0;
            const float weights[4] = {(1 - ly) * (1 - lx), (1 - ly) * lx,
                                      ly * (1 - lx), ly * lx};
            float taps[4] = {};
            for (int j = 0; j < 4; ++j)
            {
                const int xx = x0 + (j & 1), yy = y0 + (j >> 1);
                if (xx >= 0 && xx < src_w && yy >= 0 && yy < src_h)
                    taps[j] = src[static_cast<std::size_t>(yy) * src_w + xx];
            }
            const float value = weights[0] * taps[0] + weights[1] * taps[1] +
                                weights[2] * taps[2] + weights[3] * taps[3];
            // This is a RAW-mask threshold, not sigmoid(mask) > 0.5.
            out[dx] = value > 0.5f ? 255 : 0;
        }
    }
}
} // namespace sam3::trt
