#pragma once

#ifndef MODEL_TEXT_HPP__
#define MODEL_TEXT_HPP__

#include "infer/aclmodel.hpp"
#include <cstdint>
#include <string>
#include <vector>

/**
 * @brief SAM3 Text Encoder
 *
 * 输入：input_ids [1,32], attention_mask [1,32]
 * 输出：text_features [1,32,256], text_mask [1,32]
 *
 * 输出缓存在 NPU 显存中，生命周期跟随 TextModel 对象。
 */
class TextModel : public AclModel
{
  public:
    TextModel() = default;
    ~TextModel() override = default;

    /**
     * @brief 执行 text encoder 推理
     * @param input_ids      长度必须为 32
     * @param attention_mask 长度必须为 32
     * @return ACL_SUCCESS 成功
     */
    aclError encode(const std::vector<int64_t>& input_ids, const std::vector<int64_t>& attention_mask);

    void* text_features_ptr() const;
    void* text_mask_ptr() const;

    size_t text_features_size() const;
    size_t text_mask_size() const;
};

#endif // MODEL_TEXT_HPP__
