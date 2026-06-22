#pragma once

#ifndef MODEL_DECODER_HPP__
#define MODEL_DECODER_HPP__

#include "infer/aclmodel.hpp"
#include <array>

/**
 * @brief SAM3 Decoder
 *
 * 输入 0~2：来自 VisionModel 的 fpn_feat_0/1/2
 * 输入 3  ：来自 fpn_pos_2_constant.npy 的 fpn_pos_2
 * 输入 4~5：来自 TextModel 的 prompt_features 与 prompt_mask
 * 输出：pred_masks, pred_boxes, pred_logits, presence_logits
 *
 * decode 时直接使用外部显存指针构造输入 dataset，避免 D2D 拷贝。
 */
class DecoderModel : public AclModel
{
  public:
    DecoderModel() = default;
    ~DecoderModel() override = default;

    /**
     * @brief 执行 decoder 推理
     * @param vision_features     3 个 vision 输出的 device 指针
     * @param vision_sizes        对应的字节大小
     * @param fpn_pos_2           fpn_pos_2 的 device 指针
     * @param fpn_pos_2_size      字节大小
     * @param text_features       text_features 的 device 指针
     * @param text_features_size  字节大小
     * @param text_mask           text_mask 的 device 指针
     * @param text_mask_size      字节大小
     * @return ACL_SUCCESS 成功
     */
    aclError decode(const std::array<void*, 3>& vision_features,
                    const std::array<size_t, 3>& vision_sizes,
                    void* fpn_pos_2,
                    size_t fpn_pos_2_size,
                    void* text_features,
                    size_t text_features_size,
                    void* text_mask,
                    size_t text_mask_size);

    void* pred_masks_ptr() const;
    void* pred_boxes_ptr() const;
    void* pred_logits_ptr() const;
    void* presence_logits_ptr() const;
};

#endif // MODEL_DECODER_HPP__
