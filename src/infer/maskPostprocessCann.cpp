#include "infer/maskPostprocessCann.hpp"
#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>

static double now_ms()
{
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

static bool ensure_device_buffer(void*& ptr, size_t& cap, size_t needed)
{
    if (needed <= cap && ptr != nullptr)
    {
        return true;
    }
    if (ptr != nullptr)
    {
        aclrtFree(ptr);
        ptr = nullptr;
    }
    
    // 优化：避免微小的尺寸增长频繁触发重新申请，采用 1.5 倍增长策略
    size_t alloc_size = std::max(needed, cap * 3 / 2);
    // 设定一个初始的最小分配容量（例如 16 个 Mask 对应的字节数）
    const size_t MIN_ALLOC = 16ULL * 288 * 288 * sizeof(float);
    alloc_size = std::max(alloc_size, MIN_ALLOC);

    cap = 0;
    aclError ret = aclrtMalloc(&ptr, alloc_size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS)
    {
        std::cerr << "Mask postprocess malloc " << alloc_size << " bytes failed: " << ret << std::endl;
        ptr = nullptr;
        return false;
    }
    cap = alloc_size;
    return true;
}

MaskPostprocessCann::MaskPostprocessCann()
{
}

MaskPostprocessCann::~MaskPostprocessCann()
{
    if (stream_ != nullptr)
    {
        // 销毁 Stream 前，确保所有未完成的异步任务全部安全退出
        aclrtSynchronizeStream(stream_);
        aclrtDestroyStream(stream_);
        stream_      = nullptr;
        initialized_ = false;
    }

    if (selected_buf_ != nullptr)
    {
        aclrtFree(selected_buf_);
        selected_buf_ = nullptr;
    }
    selected_cap_ = 0;
}

bool MaskPostprocessCann::initialize()
{
    if (initialized_)
    {
        return true;
    }

    aclError ret = aclrtCreateStream(&stream_);
    if (ret != ACL_SUCCESS)
    {
        std::cerr << "Create mask postprocess stream failed: " << ret << std::endl;
        return false;
    }

    initialized_ = true;
    return true;
}

static cv::Mat warp_affine_mask(const float* src_data,
                                int src_size,
                                int dst_w, int dst_h,
                                float x1, float y1,
                                float scale_x, float scale_y)
{
    cv::Mat src(src_size, src_size, CV_32FC1, const_cast<float*>(src_data));
    cv::Mat dst(dst_h, dst_w, CV_32FC1);

    float M_data[6] = {
        scale_x, 0.0f, x1 * scale_x,
        0.0f, scale_y, y1 * scale_y
    };
    cv::Mat M(2, 3, CV_32F, M_data);

    cv::warpAffine(src, dst, M, dst.size(), cv::INTER_LINEAR,
                   cv::BORDER_CONSTANT, cv::Scalar(0.0f));

    cv::Mat binary;
    cv::threshold(dst, binary, 0.0f, 255.0f, cv::THRESH_BINARY);
    binary.convertTo(binary, CV_8UC1);
    return binary;
}

bool MaskPostprocessCann::process(void* pred_masks_buf,
                                  const std::vector<int>& indices,
                                  const std::vector<std::array<float, 4>>& boxes,
                                  int orig_h, int orig_w,
                                  std::vector<cv::Mat>& out_masks)
{
    if (!initialized_ || indices.empty() || boxes.size() != indices.size())
    {
        return false;
    }

    const int MASK_SIZE   = 288;
    const int N           = static_cast<int>(indices.size());
    const size_t single_in_bytes = 1ULL * MASK_SIZE * MASK_SIZE * sizeof(float);
    const size_t float_in_bytes  = N * single_in_bytes;

    out_masks.clear();
    out_masks.resize(N);

    double t_total0 = now_ms();

    // 1. D2D 把选中的 mask 拷贝到连续 buffer
    if (!ensure_device_buffer(selected_buf_, selected_cap_, float_in_bytes))
    {
        return false;
    }

    aclError ret;
    for (int i = 0; i < N; ++i)
    {
        int idx = indices[i];
        void* src = static_cast<char*>(pred_masks_buf) + idx * single_in_bytes;
        void* dst = static_cast<char*>(selected_buf_) + i * single_in_bytes;
        ret = aclrtMemcpyAsync(dst, single_in_bytes, src, single_in_bytes,
                               ACL_MEMCPY_DEVICE_TO_DEVICE, stream_);
        if (ret != ACL_SUCCESS)
        {
            std::cerr << "Mask postprocess D2D copy failed: " << ret << std::endl;
            return false;
        }
    }
    aclrtSynchronizeStream(stream_);
    double t_after_copy = now_ms();

    // 2. D2H 到 CPU
    std::vector<float> host_float(float_in_bytes / sizeof(float));
    ret = aclrtMemcpy(host_float.data(), float_in_bytes,
                      selected_buf_, float_in_bytes,
                      ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS)
    {
        std::cerr << "Mask postprocess D2H failed: " << ret << std::endl;
        return false;
    }
    double t_after_d2h = now_ms();

    // 3. CPU 并行 warpAffine
    const float scale_x = static_cast<float>(MASK_SIZE) / static_cast<float>(orig_w);
    const float scale_y = static_cast<float>(MASK_SIZE) / static_cast<float>(orig_h);

    // 优化：采用 OpenCV 自带的 cv::parallel_for_ 分发。
    // 这能让程序自动适配 OpenCV 全局优化线程池，彻底杜绝手动开辟 std::thread 的资源超发问题
    cv::parallel_for_(cv::Range(0, N), [&](const cv::Range& range) {
        for (int i = range.start; i < range.end; ++i)
        {
            float x1 = boxes[i][0];
            float y1 = boxes[i][1];
            float x2 = boxes[i][2];
            float y2 = boxes[i][3];

            x1 = std::max(0.0f, std::min(x1, static_cast<float>(orig_w)));
            y1 = std::max(0.0f, std::min(y1, static_cast<float>(orig_h)));
            x2 = std::max(0.0f, std::min(x2, static_cast<float>(orig_w)));
            y2 = std::max(0.0f, std::min(y2, static_cast<float>(orig_h)));

            int dst_w = static_cast<int>(x2 - x1);
            int dst_h = static_cast<int>(y2 - y1);

            const float* src_ptr = host_float.data() + i * MASK_SIZE * MASK_SIZE;

            if (dst_w <= 0 || dst_h <= 0)
            {
                out_masks[i] = cv::Mat::zeros(1, 1, CV_8UC1);
                continue;
            }

            out_masks[i] = warp_affine_mask(src_ptr, MASK_SIZE,
                                            dst_w, dst_h,
                                            x1, y1,
                                            scale_x, scale_y);
        }
    });

    double t_end = now_ms();
    std::cout << "[MaskPostprocessCann] N=" << N
              << " d2d_copy=" << (t_after_copy - t_total0)
              << " d2h=" << (t_after_d2h - t_after_copy)
              << " cpu_warp=" << (t_end - t_after_d2h)
              << " total=" << (t_end - t_total0) << " ms" << std::endl;

    return true;
}