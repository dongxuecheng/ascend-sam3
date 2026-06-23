#pragma once

#ifndef SAM3_INFER_HPP__
#define SAM3_INFER_HPP__

#include "infer/infer.hpp"
#include "infer/maskPostprocessCann.hpp"
#include "infer/modelDecoder.hpp"
#include "infer/modelText.hpp"
#include "infer/modelVision.hpp"
#include "infer/sam3type.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace sam3
{

/**
 * @brief std::vector<int64_t> 的简单哈希，用于 text-encoder 缓存 key。
 */
struct VectorInt64Hash
{
    std::size_t operator()(const std::vector<int64_t>& v) const noexcept;
};

} // namespace sam3

/**
 * @brief SAM3 端到端推理实现
 *
 * 支持两种方式提供 text 信息：
 *  1. 外部传入 input_ids/attention_mask，由 C++ 端 text-encoder.om 推理（text 输出常驻显存）。
 *  2. 外部直接传入 text_features/text_mask，跳过 text encoder。
 *
 * 单 batch 推理：同一张图片的 vision 输出只计算一次，多个 text prompt/word 反复复用。
 * fpn_pos_2 来自 fpn_pos_2_constant.npy，初始化时加载并常驻显存。
 */
class Sam3Infer : public Infer
{
  public:
    Sam3Infer(const ModelPaths& paths);
    ~Sam3Infer() override;

    /**
     * @brief 加载三个 OM 模型并加载 fpn_pos_2_constant.npy
     * @return true 成功
     */
    bool initialize();

    object::DetectionBoxArray forward(std::shared_ptr<Sam3Input> input) override;

  private:
    bool load_fpn_pos_2();
    aclError upload_external_text(const ExternalTextFeature& ext, void*& text_features_buf, void*& text_mask_buf);

    /**
     * @brief text-encoder 输出缓存条目
     */
    struct TextFeatureCacheEntry
    {
        void* text_features_buf = nullptr;
        void* text_mask_buf     = nullptr;
        size_t text_features_size = 0;
        size_t text_mask_size     = 0;
    };

    /**
     * @brief 获取 text-encoder 输出：优先读缓存，未命中则执行 text encoder 并缓存。
     * @return true 成功
     */
    bool get_or_encode_text(const TextPrompt& prompt,
                            void*& text_features_buf,
                            void*& text_mask_buf,
                            size_t& text_features_size,
                            size_t& text_mask_size);

    object::DetectionBoxArray postprocess(const cv::Mat& original_image,
                                          float confidence_threshold,
                                          bool need_mask,
                                          const std::string& class_name,
                                          void* pred_masks_buf,
                                          void* pred_boxes_buf,
                                          void* pred_logits_buf,
                                          void* presence_logits_buf);

    ModelPaths paths_;

    std::unique_ptr<VisionModel> vision_model_;
    std::unique_ptr<TextModel> text_model_;
    std::unique_ptr<DecoderModel> decoder_model_;

    // CANN aclnn 单算子实现的 mask 后处理
    MaskPostprocessCann mask_postprocess_;

    // fpn_pos_2 来自 npy，常驻显存
    void* fpn_pos_2_buf_ = nullptr;
    size_t fpn_pos_2_size_ = 0;

    // 外部 text 特征的上传缓存（避免反复分配）
    void* ext_text_features_buf_ = nullptr;
    void* ext_text_mask_buf_     = nullptr;
    size_t ext_text_features_size_ = 0;
    size_t ext_text_mask_size_     = 0;

    // text-encoder 输出缓存：key 为 input_ids + attention_mask 拼接后的 64 个 int64
    std::unordered_map<std::vector<int64_t>, TextFeatureCacheEntry, sam3::VectorInt64Hash> text_feature_cache_;
};

#endif // SAM3_INFER_HPP__
