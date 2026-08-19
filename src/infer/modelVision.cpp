#include "infer/modelVision.hpp"
#include <chrono>
#include <iostream>

static double now_ms()
{
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

aclError VisionModel::encode(const cv::Mat& image)
{
    if (image.empty())
    {
        std::cerr << "VisionModel encode got empty image" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }

    aclError ret;
    double t0 = now_ms();
    ret = preprocess_bgr(image);

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

    std::cout << "[Time] Vision preprocess: " << (t1 - t0) << " ms, inference: " << (t2 - t1)
              << " ms, total: " << (t2 - t0) << " ms" << std::endl;
    return ACL_SUCCESS;
}

void* VisionModel::feature_ptr(size_t idx) const
{
    return output_buffer(idx);
}

aclError VisionModel::preprocess_bgr(const cv::Mat& image)
{
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(input_w_, input_h_));
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

    CHECK_ACL(aclrtMemcpy(input_buffer(0), input_size(0), resized.data, data_size,
                          ACL_MEMCPY_HOST_TO_DEVICE));
    return ACL_SUCCESS;
}
