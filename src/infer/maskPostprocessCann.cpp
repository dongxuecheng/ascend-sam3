#include "infer/maskPostprocessCann.hpp"
#include <iostream>

MaskPostprocessCann::MaskPostprocessCann()
{
}

MaskPostprocessCann::~MaskPostprocessCann()
{
    if (stream_ != nullptr)
    {
        aclrtDestroyStream(stream_);
        stream_      = nullptr;
        initialized_ = false;
    }
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

bool MaskPostprocessCann::process(void* /*pred_masks_buf*/,
                                  const std::vector<int>& /*indices*/,
                                  int /*orig_h*/, int /*orig_w*/,
                                  std::vector<cv::Mat>& /*out_masks*/)
{
    // 当前 Ascend310P + CANN 8.2.RC1 环境下 aclnnUpsampleNearest2d 等 resize 算子
    // 找不到 NPU kernel（561003）或出现内部空指针（561103），直接返回 false，
    // 让上层调用者走 CPU/OpenCV fallback。后续 CANN 版本支持后可在此恢复 NPU 实现。
    return false;
}
