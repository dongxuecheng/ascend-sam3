#ifndef SAM3_HPP__
#define SAM3_HPP__

#include <cstdint>
#include <optional>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

struct TextPrompt
{
    std::string text;                    // 原始文本 prompt，用于结果标注
    std::vector<int64_t> input_ids;      // 长度 32
    std::vector<int64_t> attention_mask; // 长度 32
};

/**
 * @brief 由外部（如 Python）预先算好的 text encoder 输出
 *
 * text_features: [1, 32, 256] 的 float32 数据，按 NCHW/NHWC 展平均可，
 *                这里直接按一维 float vector 传入，大小应为 8192。
 * text_mask:     [1, 32] 的 bool/uint8 数据，大小应为 32。
 */
struct ExternalTextFeature
{
    std::string class_name;           // 用于结果标注的类别名
    std::vector<float> text_features; // size = 1 * 32 * 256
    std::vector<uint8_t> text_mask;   // size = 1 * 32
};

struct Sam3Input
{
    float confidence_threshold = 0.3f;
    bool need_mask             = true; // 是否需要输出分割掩码；false 时跳过 mask 解码，仅返回检测框
    cv::Mat image;

    // 方式一：C++ 内部使用 text-encoder.om 推理（调用方提供 input_ids/attention_mask）
    std::vector<TextPrompt> text_prompts;

    // 方式二：外部传入 text encoder 输出，优先级高于 text_prompts
    std::vector<ExternalTextFeature> external_text_features;
};

#endif
