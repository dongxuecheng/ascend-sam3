#include "infer/modelVision.hpp"
#include "common/trtSampling.hpp"
#include <chrono>
#include <iostream>

static double now_ms()
{
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

aclError VisionModel::encode(const cv::Mat& image)
{
    return encode_impl(image, nullptr);
}

aclError VisionModel::encode_crop(const cv::Mat& image, const cv::Rect& crop)
{
    if (crop.x < 0 || crop.y < 0 || crop.width <= 0 || crop.height <= 0 ||
        crop.x > image.cols || crop.y > image.rows ||
        crop.width > image.cols - crop.x || crop.height > image.rows - crop.y)
        return ACL_ERROR_INVALID_PARAM;
    return encode_impl(image, &crop);
}

aclError VisionModel::encode_impl(const cv::Mat& image, const cv::Rect* crop)
{
    if (image.empty() || image.type() != CV_8UC3)
    {
        std::cerr << "VisionModel requires a non-empty BGR uint8 image" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }

    aclError ret;
    timings_ = Timings{};
    double t0 = now_ms();
    ret = preprocess_bgr(image, crop);

    if (ret != ACL_SUCCESS)
    {
        return ret;
    }
    double t1 = now_ms();

    ret = execute();
    if (ret != ACL_SUCCESS)
    {
        std::cerr << "VisionModel execute failed" << std::endl;
        return ret;
    }

    synchronize();
    double t2 = now_ms();
    timings_.inference_ms = t2 - t1;
    timings_.total_ms = t2 - t0;
    if (timing_log_)
        std::cout << "[Time] Vision preprocess: " << (t1 - t0) << " ms, inference: " << (t2 - t1)
                  << " ms, total: " << (t2 - t0) << " ms" << std::endl;
    return ACL_SUCCESS;
}

void* VisionModel::feature_ptr(size_t idx) const
{
    return output_buffer(idx);
}

aclError VisionModel::preprocess_bgr(const cv::Mat& image, const cv::Rect* crop)
{
    const double started = now_ms();
    cv::Mat resized(input_h_, input_w_, CV_8UC3);
    const auto matrix = crop ?
        sam3::trt::crop_inverse(crop->x, crop->y, crop->width, crop->height, input_w_, input_h_) :
        sam3::trt::resize_inverse(image.cols, image.rows, input_w_, input_h_);
    // Do not use cv::warpAffine: its interpolation table quantizes fractions.
    // Use float bilinear weights, uint8 rounding and border=114 as TRT does.
    cv::parallel_for_(cv::Range(0, input_h_), [&](const cv::Range& rows) {
        sam3::trt::warp_bgr_rows(image.data, image.step[0], image.cols, image.rows,
                               resized.data, resized.step[0], input_w_, matrix, rows.start, rows.end);
    });
    size_t data_size = resized.total() * resized.elemSize(); // 1008 * 1008 * 3

    // 当前预处理依赖 OM 中的静态 AIPP，外部输入必须是 RGB888/BGR uint8。
    // 若误用了未插入 AIPP 的 float32 OM，立即报错，避免只复制四分之一数据后
    // 得到看似成功但数值错误的推理结果。
    if (input_size(0) != data_size)
    {
        std::cerr << "Vision input size mismatch: OM expects " << input_size(0)
                  << " bytes, but AIPP uint8 input is " << data_size << " bytes" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }

    const double sampled = now_ms();
    const aclError ret = aclrtMemcpy(input_buffer(0), input_size(0), resized.data, data_size,
                                    ACL_MEMCPY_HOST_TO_DEVICE);
    timings_.sampling_ms = sampled - started;
    timings_.upload_ms = now_ms() - sampled;
    if (ret != ACL_SUCCESS) return ret;
    return ACL_SUCCESS;
}
