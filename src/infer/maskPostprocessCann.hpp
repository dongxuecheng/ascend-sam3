#pragma once

#ifndef MASK_POSTPROCESS_CANN_HPP__
#define MASK_POSTPROCESS_CANN_HPP__

#include <acl/acl.h>
#include <opencv2/opencv.hpp>
#include <array>
#include <cstdint>
#include <vector>

/**
 * @brief 使用 ACL 内存拷贝与 OpenCV 对 SAM3 decoder mask 进行批量后处理。
 *
 * 处理流程：
 *   1. D2D memcpy 把选中的 N 张 mask 从 [1,200,288,288] 拷贝到连续 buffer -> [1,N,288,288]
 *   2. D2H 到 CPU
 *   3. OpenCV 按目标框裁剪、插值、二值化
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
     * @brief 批量解码 mask，输出每张 mask 裁剪到目标框尺寸。
     *
     * 实现方式：
     * 1. D2D 把选中的 mask 从 decoder 输出 [1,200,288,288] 拷贝到连续 buffer。
     * 2. D2H 到 CPU。
     * 3. 对每张 288x288 float mask，按目标框在原图的比例裁剪并双线性缩放到框尺寸，
     *    然后阈值化为 0/255 uint8。
     *
     * @param pred_masks_buf decoder 输出的 pred_masks，形状 [1, 200, 288, 288]，float32
     * @param indices 选中的 mask 索引，范围 [0, 200)
     * @param boxes 每张 mask 对应的目标框，顺序与 indices 一致；绝对坐标 [x1, y1, x2, y2]
     * @param orig_h 原图高度
     * @param orig_w 原图宽度
     * @param out_masks 输出：每张裁剪并 resize 到目标框尺寸的二值 mask（0/255）
     * @return true 成功
     */
    bool process(void* pred_masks_buf,
                 const std::vector<int>& indices,
                 const std::vector<std::array<float, 4>>& boxes,
                 int orig_h, int orig_w,
                 std::vector<cv::Mat>& out_masks);

  private:
    aclrtStream stream_ = nullptr;
    bool initialized_   = false;

    // device 缓存，避免每次 process 重复 aclrtMalloc/free
    void* selected_buf_  = nullptr;
    size_t selected_cap_  = 0;
};

#endif // MASK_POSTPROCESS_CANN_HPP__
