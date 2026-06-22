#pragma once

#ifndef MODEL_VISION_HPP__
#define MODEL_VISION_HPP__

#include "infer/aclmodel.hpp"
#include <opencv2/opencv.hpp>
#include <string>

/**
 * @brief SAM3 Vision Encoder
 *
 * 输入：images [1,3,1008,1008] float32（也兼容 YUV420SP 带 AIPP 的 OM）
 * 输出：fpn_feat_0、fpn_feat_1、fpn_feat_2
 *
 * 注意：fpn_pos_2 现在由外部 fpn_pos_2_constant.npy 提供，不再来自 Vision Encoder。
 */
class VisionModel : public AclModel
{
  public:
    VisionModel() = default;
    ~VisionModel() override = default;

    /**
     * @brief 对图像进行预处理并执行 vision encoder 推理
     * @param image 任意尺寸的 BGR 图像
     * @return ACL_SUCCESS 成功
     */
    aclError encode(const cv::Mat& image);

    /**
     * @brief 获取第 idx 个输出的 NPU 显存地址
     *        有效索引为 0(fpn_feat_0)、1(fpn_feat_1)、2(fpn_feat_2)。
     */
    void* feature_ptr(size_t idx) const;

  private:
    aclError preprocess_bgr(const cv::Mat& image);
    int input_h_ = 1008;
    int input_w_ = 1008;
};

#endif // MODEL_VISION_HPP__
