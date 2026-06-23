#pragma once

#ifndef MASK_POSTPROCESS_CANN_HPP__
#define MASK_POSTPROCESS_CANN_HPP__

#include <acl/acl.h>
#include <aclnnop/aclnn_cast.h>
#include <aclnnop/aclnn_gt_scalar.h>
#include <aclnnop/aclnn_upsample_nearest_2d.h>
#include <opencv2/opencv.hpp>
#include <cstdint>
#include <vector>

/**
 * @brief 使用 CANN aclnn 单算子对 SAM3 decoder 输出的 mask logit 进行批量后处理。
 *
 * 处理流程：
 *   1. D2D memcpy 把选中的 N 张 mask 从 [1,200,288,288] 拷贝到连续 buffer -> [1,N,288,288]
 *   2. aclnnGtScalar(0.0) 二值化 -> bool
 *   3. aclnnCast -> uint8
 *   4. aclnnUpsampleNearest2d resize 到原图尺寸 -> [1,N,orig_h,orig_w]
 *   5. D2H 并拆分为 N 张 cv::Mat
 *
 * 调用方负责保证 pred_masks_buf 为有效的 NPU device 指针，且 decoder 已完成同步。
 */
class MaskPostprocessCann
{
  public:
    MaskPostprocessCann();
    ~MaskPostprocessCann();

    /**
     * @brief 初始化内部 aclrtStream。
     * @return true 成功
     */
    bool initialize();

    /**
     * @brief 批量解码 mask。
     * @param pred_masks_buf decoder 输出的 pred_masks，形状 [1, 200, 288, 288]，float32
     * @param indices 选中的 mask 索引，范围 [0, 200)
     * @param orig_h 原图高度
     * @param orig_w 原图宽度
     * @param out_masks 输出：每张 resize 后的完整原图尺寸二值 mask（0/1）
     * @return true 成功
     */
    bool process(void* pred_masks_buf,
                 const std::vector<int>& indices,
                 int orig_h, int orig_w,
                 std::vector<cv::Mat>& out_masks);

  private:
    aclrtStream stream_ = nullptr;
    bool initialized_   = false;
};

#endif // MASK_POSTPROCESS_CANN_HPP__
